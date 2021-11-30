# Do glossary

import os
import pandas as pd
import numpy as np
import textwrap
import sys
import praw
import prawcore

import re

def tex_escape(text):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
        https://stackoverflow.com/a/25875504/764272
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key = lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)

from commembed.jupyter import *
embedding = load_embedding('reddit', 'master')
dimen_list = dimens.load_dimen_list('final')
scores = dimens.score_embedding(embedding, dimen_list)

scores = scores.apply(lambda x: (x - np.mean(x)) / np.std(x), axis=0) \
    .apply(lambda x: np.sign(x) * np.floor(np.abs(x)), axis=0) \
    .apply(lambda x: np.maximum(-3, np.minimum(3, x)))

root = '/u/walleris/research/commembed/paper_resources'
fnames = [root + '/' + fname for fname in os.listdir(root) if fname.startswith('glossary_') and not (fname.startswith("glossary_fig_A"))]
print("Glossary includes\n", file=sys.stderr)
dfs = []
for fname in fnames:
    try:
        df = pd.read_csv(fname, header=None).values.flatten()
        
        df = df[~pd.isnull(df)]
        print("\t%s (%d)" % (fname, len(df)), file=sys.stderr)
        dfs.append(df)
    except Exception as e:
        continue # Great code
subs = np.concatenate(dfs).tolist()
subs = sorted(list(set(subs)), key=str.casefold)
subs = pd.DataFrame(index=subs)
subs["first_letter"] = subs.index.str.slice(0, 1).str.upper()

subs.loc[~subs["first_letter"].str.match("[A-Z]"), "first_letter"] = "0-9"
subs

additional_descriptions_path = os.path.join(os.path.dirname(__file__), "additional_descriptions.csv")
additional_descriptions = pd.read_csv(additional_descriptions_path, header=0, index_col="subreddit")

reddit = praw.Reddit(client_id=None,
        client_secret=None,
        user_agent="walleris@cs.toronto.edu community embeddings script")

def get_additional_description(comm):
    # download if missing
    if comm in additional_descriptions.index:
        return additional_descriptions.loc[comm, "description"]
    else:
        print("Downloading description for %s" % comm, file=sys.stderr)

        subreddit = reddit.subreddit(comm)
        desc = subreddit.public_description or subreddit.title

        additional_descriptions.loc[comm] = [desc]

        return desc

def generate_glossary(comms):
    
    to_print = embedding.metadata.loc[comms][['description']].join(scores[['age', 'gender', 'partisan']])
    
    
    result = []
    i = 0
    for comm, row in to_print.iterrows():
        
        desc = row["description"]
        desc = desc if desc == desc else get_additional_description(comm) # nan check
        
        desc = desc.replace('#', ' ').replace('*', ' ') # remove markdown

        # remove links
        desc = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", desc)

        # remove non latin characters
        desc = re.sub(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', '', desc) 

        trunc_desc = textwrap.fill(desc, 130).split("\n")[0]
        if trunc_desc != desc:
            trunc_desc += "..."
            
        comm = comm.replace('_', "\\_")
        trunc_desc = tex_escape(trunc_desc)
        trunc_desc = trunc_desc if trunc_desc != "Subreddit forbidden" else "\\textit{Community banned or deleted prior to publication}"
        trunc_desc = trunc_desc if trunc_desc != "Subreddit no longer exists" else "\\textit{Community banned or deleted prior to publication}"

        fmt_score = lambda x: str(int(x)).replace('-','n')
        age_score = "age_" + fmt_score(row["age"])
        gender_score = "gender_" + fmt_score(row["gender"])
        partisan_score = "partisan_" + fmt_score(row["partisan"])
        
        result.append("\\glossarytitle{%s}{gstd_%s}{gstd_%s}{gstd_%s} \\newline \\glossarydesc{%s}" % (comm, age_score, gender_score, partisan_score, trunc_desc)
                      + ("&" if (i%2)==1 else "\\\\"))
        
        i += 1
        
    result = '\n'.join(result)

    if (i%2)==0:
        result += "\\\\"
    
    print(result)
    
for first_letter, rows in subs.groupby("first_letter"):
    print("\\glossaryheader{%s} &" % first_letter)
    generate_glossary(rows.index.tolist())
    

# Save additional descriptions
additional_descriptions.to_csv(additional_descriptions_path)
