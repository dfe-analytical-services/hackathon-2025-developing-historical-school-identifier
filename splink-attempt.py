# pip.main(['install', 'splink'])
# pip.main(['install', 'pyspark'])
# pip.main(['install', 'duckdb'])
# pip.main(['install', 'pyarrow'])
# pip.main(['install', 'pandas'])

from itertools import count
import pip
from requests import head
import splink
import pyspark 
import pandas as pd
import re 
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


# read gias data
gias = pd.read_csv('Data/gias_data2024-03-01_2024-09-01_28.csv')


print(duckdb.from_df(gias).df().head())

# clean nulls 

for col_name in gias.columns:
    gias[col_name] = gias[col_name].replace(
        to_replace=[r"^\s*$", r"^NA$", r"^NA NA$", r"^na$", r"^NaN$", r"^nan$", r"^N/A$", r"^n/a$"],
        value=pd.NA,
        regex=True
    )

# issue with mixed data types in laestab
gias['laestab'] = gias['laestab'].astype(str)

    
def cleanse_names(series: pd.Series) -> pd.Series:
    """
    Clean text columns similar to your Spark UDF logic.
    """
    # lowercase
    cleaned = series.str.lower()

    # remove special characters (keep only letters, digits, space)
    cleaned = cleaned.str.replace(r"[^a-z0-9 ]", "", regex=True)

    # normalize whitespace
    cleaned = cleaned.str.strip().str.replace(r"\s+", " ", regex=True)

    # replace empty strings with None/NaN
    cleaned = cleaned.replace("", pd.NA)

    return cleaned


cols_to_clean = [
    "heads_name",
    "establishment_name",
    "previous_establishment_number",
    "trusts_name",
]

for c in cols_to_clean:
    gias[c] = cleanse_names(gias[c].astype(str))


    # Ensure all columns are strings for Splink
    for col in gias.columns:
        gias[col] = gias[col].astype(str)

# Drop duplicates ignoring 'gias_date' column
cols_to_check = [c for c in gias.columns if c != "gias_date"]


gias = gias.drop(columns=['Unnamed: 0'])
subset = gias.columns.difference(['gias_date'])

gias = gias.drop_duplicates(subset=subset)


gias["unique_id"] = range(1, len(gias) + 1)


print(gias.columns)

print(len(gias))

# gias_1 = gias.copy()

# gias_2 = gias.copy()    

# # Add unique_id (sequential)
# gias_1["unique_id"] = range(1, len(gias_1) + 1)
# gias_2["unique_id"] = range(1, len(gias_2) + 1)

# # toArrow makes it available to DuckDB
# # This gets loaded in memory so should already be small at this point.
# gias_1_data_arw = pa.Table.from_pandas(gias_1)
# gias_2_data_arw = pa.Table.from_pandas(gias_2)


completeness_chart(
    gias,
    db_api=db_api)


profile_columns(gias, db_api=db_api, column_expressions=["ukprn"])"])

profile_columns(gias, db_api=db_api, column_expressions=["heads_name"])



# custom comparison for full_name

headteacher_name_comparison = CustomComparison(
    output_column_name="heads_name",
    comparison_levels=[
        cll.NullLevel("heads_name"),
        cll.ExactMatchLevel("heads_name").configure(tf_adjustment_column="heads_name"),
        cll.JaroWinklerLevel("heads_name", 0.9).configure(tf_adjustment_column="heads_name"),
        cll.ElseLevel(),
    ],
)

# from splink.comparison_library import CustomComparison
# from splink.comparison_level_library import comparison_level_library as cll


northing_easting_comparison = CustomComparison(
    output_column_name="northing_easting_distance",
    comparison_levels=[
        cll.NullLevel("easting", "northing"),  # level 0: nulls
        cll.ExactMatchLevel("easting", "northing"),

        {
            "sql_condition": """
                (a.easting IS NOT NULL AND a.northing IS NOT NULL AND
                 b.easting IS NOT NULL AND b.northing IS NOT NULL AND
                 ((a.easting - b.easting)*(a.easting - b.easting) +
                  (a.northing - b.northing)*(a.northing - b.northing)) < 10000)
            """,
            "tf_adjustment_column": "easting"  # close?
        },
        cll.ElseLevel()  # everything else
    ]
)


#print(headteacher_name_comparison.get_comparison("duckdb").human_readable_description)


settings = SettingsCreator(
    link_type="dedupe_only",
    unique_id_column_name="unique_id",
    # probability_two_random_records_match=1e-6,  # very small
    blocking_rules_to_generate_predictions=blocking_rules_link,
    comparisons=[
        cl.ExactMatch("urn"),
        cl.ExactMatch("laestab"),
        cl.PostcodeComparison("postcode"),
        cl.NameComparison("establishment_name"),
     #   headteacher_name_comparison,
      #  northing_easting_comparison
    ],
    retain_intermediate_calculation_columns=True,
)

linker = Linker(
    gias,
    settings,
    db_api=db_api,
    validate_settings=True,
)

# I dont seem able to train the model with this as too many matches on urn?

linker.training.estimate_probability_two_random_records_match(
    [
        #  block_on("establishment_name"),
        block_on("urn"),
       # block_on("laestab"),
        #block_on("postcode"),
       # block_on("heads_name"),
      #  block_on("trusts_name"),
     #   block_on("northing", "easting"),
    ],
    recall=0.95,
)

linker.training.estimate_u_using_random_sampling(max_pairs=1e10)


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

linker.training.estimate_parameters_using_expectation_maximisation(
     blocking_rule=block_on("postcode"),
 )

linker.visualisations.parameter_estimate_comparisons_chart()


linker.visualisations.match_weights_chart()

linker.evaluation.unlinkables_chart()

df_predict = linker.inference.predict()

df_e = df_predict.as_pandas_dataframe()

df_e = df_e.sort_values(by="match_probability", ascending=False)


filtered_df = df_e[df_e["match_probability"] > 0.6]

print(f"Number of rows in that match: {len(filtered_df)}")

print(f"Match percentage: {len(filtered_df)/len(df_e)}")