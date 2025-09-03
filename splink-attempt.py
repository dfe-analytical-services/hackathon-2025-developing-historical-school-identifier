# pip.main(['install', 'splink'])
# pip.main(['install', 'pyspark'])
# pip.main(['install', 'duckdb'])
# pip.main(['install', 'pyarrow'])
# pip.main(['install', 'pandas'])

import pip
import splink
import pyspark 

from pyspark.sql.functions import col, monotonically_increasing_id, expr, concat_ws, lit, split
from pyspark.sql.functions import col, date_format, lower, regexp_replace, expr, trim, when, greatest
from pyspark.sql.functions import concat_ws, col, split, slice, element_at, size, to_date, coalesce
from pyspark.sql.types import StringType
from pyspark.sql import DataFrame
from pyspark.sql.functions import expr
from pyspark.sql.functions import transform, sequence
import pandas as pd
import pyarrow as pa


# SPINK setup
import splink.comparison_library as cl
import splink.comparison_level_library as cll
from splink.exploratory import profile_columns
from splink.comparison_library import CustomComparison
import duckdb, os, tempfile
import sys
from splink.blocking_analysis import (
    cumulative_comparisons_to_be_scored_from_blocking_rules_chart,
)
from splink import DuckDBAPI, Linker, SettingsCreator, block_on
from splink.exploratory import completeness_chart
import csv

# Set up DuckDB in memory
# In theory we can set this to a path on the local drive, but it will be slower
con = duckdb.connect(":memory:")

# Set up temporary dir for disk spilling.
spill_dir = tempfile.mkdtemp(prefix="duckdb_spill_")
con.execute("SET memory_limit = '100GB';")  # synonyms: max_memory / memory_limit
con.execute(f"SET temp_directory = '{spill_dir}';")
con.execute("SET max_temp_directory_size = '200GB';")

# This gets used across various splink functions
db_api = DuckDBAPI(con)


def remove_special_chars(column):
    cleaned = regexp_replace(lower(column), "[^a-zA-Z0-9]", "")
    return when(cleaned == "", None).otherwise(cleaned) 


def cleanse_names(column):
    cleaned = lower(column)
    cleaned = regexp_replace(cleaned, "[^a-zA-Z0-9 ]", "")  # remove special chars
    cleaned = regexp_replace(trim(cleaned), r"\s+", " ")  # normalize whitespace
    return when(cleaned == "", None).otherwise(cleaned)  # nullify empty strings

gias = pd.read_csv('gias_data.csv')

print(duckdb.from_df(gias).head())


gias_data_edited = (
    gias.withColumn("last_name", cleanse_names(col("last_name")))
    .withColumn(
        "last_last_name",
        element_at(
            split(col("last_name"), " "), -1
        ),  # take last word as last name so double barrel dropped if not hyphenated
    )
    .withColumn("middle_names", cleanse_names(col("middle_names")))
    .withColumn("first_names", cleanse_names(col("first_names")))
    .withColumn("first_name", element_at(split(col("first_names"), " "), 1))
    .withColumn(
        "middle_names",
        when(
            col("middle_names")
            == concat_ws(" ", slice(split(col("first_names"), " "), 2, 100)),
            col("middle_names"),
        ).otherwise(
            concat_ws(
                " ", col("middle_names"), slice(split(col("first_names"), " "), 2, 100)
            )
        ),
    )
    .withColumn(
        "middle_names",
        when(trim(col("middle_names")) == "", None).otherwise(
            trim(col("middle_names"))
        ),
    )
    .withColumn(
        "full_name",
        concat_ws(" ", col("first_name"), col("middle_names"), col("last_name")),
    )
).withColumn(
    "unique_id", monotonically_increasing_id()
)


# toArrow makes it available to DuckDB
# This gets loaded in memory so should already be small at this point.
gias_data_arw = gias_data_edited.toArrow()


completeness_chart(
    gias_data_arw,
    db_api=db_api)


profile_columns(gias_data_arw, db_api, column_expressions=["first_name","last_last_name","full_name"])


blocking_rules_dedupe = [
    block_on("urn"),
    block_on("la_estab"),
    block_on("headteacher_name"),
    block_on("ukprn"),
    block_on("postcode"),
    block_on("phase")
]

cumulative_comparisons_to_be_scored_from_blocking_rules_chart(
    table_or_tables=gias_data_arw,
    blocking_rules=blocking_rules_dedupe,
    db_api=db_api,
    link_type="link_only",
)


# custom comparison for full_name

headteacher_name_comparison = CustomComparison(
    output_column_name="headteacher_name",
    comparison_levels=[
        cll.NullLevel("headteacher_name"),
        cll.ExactMatchLevel("headteacher_name").configure(tf_adjustment_column="headteacher_name"),
        cll.JaroWinklerLevel("headteacher_name", 0.9).configure(tf_adjustment_column="headteacher_name"),
        cll.ElseLevel(),
    ],
)

northing_easting_comparison = CustomComparison(
  output_column_name="northing_easting",
  comparison_levels=[
    cll.NullLevel("easting", "northing"),
    {
      "sql_condition": "(a.easting IS NOT NULL AND b.easting IS NOT NULL AND (a.easting - b.easting)*(a.easting - b.easting) < 10000)",
      "label": "Squared difference of easting < 10000",
      "tf_adjustment_column": "easting"
    },
    cll.ElseLevel(),
  ],
)

print(headteacher_name_comparison.get_comparison("duckdb").human_readable_description)


settings = SettingsCreator(
    link_type="dedupe_only",
    unique_id_column_name="unique_id",
    blocking_rules_to_generate_predictions= blocking_rules_dedupe,
    comparisons=[
        cl.ExactMatch("urn"),
        cl.ExactMatch("ukprn"),
        cl.PostcodeComparison("postcode"),
        cl.NameComparison("establishment_name"),
        headteacher_name_comparison,
        cl.ExactMatch("establishment_status_name")
    ],
    retain_intermediate_calculation_columns=True,
)

linker = Linker(
    gias_data_arw,
    settings,
    db_api=db_api,
    validate_settings=True,
)



linker.training.estimate_probability_two_random_records_match(
    [
        block_on("urn","establishment_name")
    ],
    recall=0.96,
)

linker.training.estimate_u_using_random_sampling(max_pairs=1e8)


training_blocking_rule = block_on("urn")
training_session_names = (
    linker.training.estimate_parameters_using_expectation_maximisation(
        training_blocking_rule, estimate_without_term_frequencies=True
    )
)

linker.training.estimate_parameters_using_expectation_maximisation(
    blocking_rule=block_on("establishment_name"),
)

linker.training.estimate_parameters_using_expectation_maximisation(
    blocking_rule=block_on("laestab"),
)

linker.visualisations.parameter_estimate_comparisons_chart()