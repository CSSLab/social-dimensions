
import pandas as pd
import os
import re
import json
import numpy as np

data_dir = os.path.join(os.path.dirname(__file__), '../../../data')


class Validation:
    def __init__(self, embedding):
        pass

    def data(self):
        return {}


def map_embedding_to_census_places(embedding, places):

    with open(os.path.join(data_dir, 'labels', 'us-census-metro-areas.json')) as json_file:
        return json.load(json_file)


class CensusMSAIncome(Validation):
    def __init__(self, embedding):
        data = pd.read_csv(os.path.join(data_dir, 'us-acs-2017-5yr-median-household-income-by-msa.csv'),index_col=2)
        data = data.drop("Geography")

        mapping = map_embedding_to_census_places(embedding, data.index)

        data['subreddit'] = [mapping.get(d, "") for d in data.index]
        data = data.reset_index().set_index('subreddit')

        columns = {
            "GEO.display-label": 'label',
            "HD01_VD01": "income"
        }
        data = data.rename(columns=columns)[list(columns.values())]

        data['valid'] = [(d in embedding.vectors.index) for d in data.index]

        self.df = data

    def data(self, include_invalid=False):
        if not include_invalid:
            valid_df = self.df[self.df['valid']]
        else:
            valid_df = self.df

        return {
            "uscensus_income": pd.DataFrame({
                "label": valid_df["label"],
                "y": valid_df["income"].astype(float)
            }, index=valid_df.index).sort_values("y")
        }



class ElectionResults(Validation):
    def __init__(self, embedding):

        with open(os.path.join(data_dir, 'labels', 'us-census-metro-areas.json')) as json_file:
            self.metros = json.load(json_file)

        msa_to_county = pd.read_csv(os.path.join(data_dir, "us_msa_to_county.csv"),  encoding = "ISO-8859-1",header=2)
        msa_to_county = msa_to_county[msa_to_county["Metropolitan/Micropolitan Statistical Area"] == "Metropolitan Statistical Area"]
        msa_to_county["CBSA Title"] += " Metro Area"
        msa_to_county = msa_to_county[msa_to_county["CBSA Title"].isin(list(self.metros.keys()))]
        msa_to_county["FIPS"] = msa_to_county.apply(lambda r: "%02d%03d" % (r["FIPS State Code"], r["FIPS County Code"]), axis=1).astype(int)
        msa_to_county = msa_to_county[["CBSA Title", "FIPS", "County/County Equivalent"]]
        self.msa_to_county = msa_to_county.set_index("FIPS")
        
        election_results = pd.read_csv(os.path.join(data_dir, "us_election_results_president_county.csv"))
        election_results = election_results[~np.isnan(election_results["FIPS"])]
        election_results["FIPS"] = election_results["FIPS"].astype(int)

        def aggregate_results(rows):
            
            total = rows["totalvotes"].iloc[0]
            rows = rows.set_index("party")
        
            return pd.Series({
                "state": rows["state"].iloc[0],
                "county": rows["county"].iloc[0],
                "votes_dem": rows.loc["democrat", "candidatevotes"],
                "votes_rep": rows.loc["republican", "candidatevotes"],
                "votes_total": total
            })

        self.election_results = election_results.groupby(["FIPS", "year"]).apply(aggregate_results).reset_index()
        
    def data(self, include_invalid=False):

        result = {}

        for election_year in [2008, 2012, 2016]:
            election_results = self.election_results[self.election_results["year"] == election_year].set_index("FIPS")
            results = election_results.join(self.msa_to_county, on="FIPS", rsuffix="_", how="inner").groupby("CBSA Title") \
                .agg({"votes_dem": np.sum,"votes_rep":np.sum,"votes_total":np.sum})

            results["vote_differential"] = (results["votes_rep"] -results["votes_dem"]) * 100 / results["votes_total"]

            results["subreddit"] = pd.Series(self.metros)
            results = results[~results["subreddit"].isna()]
            results = results.set_index("subreddit").rename(columns={"CBSA Title":"label","vote_differential":"y"})
            result["us_election_%d" % election_year] = results

        return result

class CensusOccupation(Validation):
    def __init__(self, embedding):

        def load_occupation_data(gender):
            data_female = pd.read_csv(os.path.join(data_dir, 'acs_occupation_%s.csv' % gender)).T
            data_female.columns = ["label", "value"]

            data_female = data_female["label"].str.split("!!", expand=True).join(data_female["value"])
            data_female = data_female[data_female[0] == "Estimate"]
            data_female.columns = [0, 1, "occupation", "value"]
            data_female = data_female.drop(columns=[0,1]).set_index("occupation", drop=True)
            return data_female
            
        occ_female = load_occupation_data("female")
        occ_male = load_occupation_data("male")

        occs = occ_female.join(occ_male, lsuffix='_f', rsuffix='_m')
        occs['value_f'] = occs['value_f'].astype(int)
        occs['value_m'] = occs['value_m'].astype(int)
        occs['pct_f'] = occs['value_f'].astype(int) / (occs['value_f'].astype(int) + occs['value_m'].astype(int))
        occs = occs.sort_values('pct_f')

        mapping = {
            "Firefighting": "Firefighters",
            "civilengineering": "Civil engineers",
            "Construction": "Construction laborers",
            "metalworking": ["Sheet metal workers", "Other metal workers and plastic workers"],
            "Carpentry": "Carpenters",
            "electricians": "Electricians",
            "Plumbing": "Plumbers, pipefitters, and steamfitters",
            "Truckers": "Driver/sales workers and truck drivers",
            "mechanics": "Automotive service technicians and mechanics",
            "farming": "Farmers, ranchers, and other agricultural managers",
            "humanresources": "Human resources workers",
            "teaching": ["Elementary and middle school teachers", "Secondary school teachers"],
            "ECEProfessionals": "Preschool and kindergarten teachers",
            "nursing": "Registered nurses",
            "Dentistry": ["Dentists", "Dental hygienists", "Dental assistants"],
            "psychotherapy": ["Marriage and family therapists", "Therapists, all other"],
            "specialed": "Special education teachers",
            "socialwork": ["Child, family, and school social workers",
                        "Mental health and substance abuse social workers",
                        "Social workers, all other"],
            "Nanny": "Childcare workers",
            "optometry": "Optometrists",
            "pharmacy": ["Pharmacists", "Pharmacy aides"],
            "librarians": "Librarians and media collections specialists",
            "Professors": "Postsecondary teachers"
        }

        to_plot = pd.DataFrame(list(mapping.items())).rename(columns={0:"subreddit", 1:"occ"}).explode("occ").join(occs, on='occ')

        to_plot = to_plot.groupby('subreddit').agg({'occ': list, 'value_f': np.sum, 'value_m': np.sum})
        to_plot['pct_f'] = to_plot['value_f'] / (to_plot['value_f'] + to_plot['value_m'])
        to_plot['y'] = to_plot['pct_f']
        self.df = to_plot

    def data(self, include_invalid=False):
        return {"us_census_occupations":self.df}

def all_validations(embedding, include_invalid=False):

    all_validation_classes = [CensusMSAIncome,ElectionResults,CensusOccupation]
    result = []

    for c in all_validation_classes:
        result.extend(c(embedding).data(include_invalid=include_invalid).items())

    return dict(result)
