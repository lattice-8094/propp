import pandas as pd
import numpy as np


def agent_attribution(tokens_df):
    tokens_df['char_att_agent'] = -1

    # condition_1 - direct verb subject
    subject_rows = tokens_df[(tokens_df["dependency_relation"] == "nsubj")
                             & (tokens_df["is_mention_head"] == 1)]
    verb_rows = tokens_df[(tokens_df["POS_tag"] == "VERB")
                          & (tokens_df["token_ID_within_document"].isin(subject_rows['syntactic_head_ID']))
                          ]

    merged_df = pd.merge(verb_rows, subject_rows, left_on='token_ID_within_document', right_on='syntactic_head_ID',
                         how='left')
    mention_head_id_list, verb_agent_head_id_list = merged_df['token_ID_within_document_y'].tolist(), merged_df[
        'token_ID_within_document_x'].tolist()
    del merged_df
    tokens_df.loc[verb_agent_head_id_list, 'char_att_agent'] = mention_head_id_list

    # condition_2 - conj synthactic relation
    verb_rows = tokens_df[(tokens_df["POS_tag"] == "VERB")
                          & (tokens_df["token_ID_within_document"].isin(subject_rows['syntactic_head_ID'].tolist()))
                          ]
    conj_verb_rows = tokens_df[(tokens_df["POS_tag"] == "VERB")
                               & (tokens_df["dependency_relation"] == "conj")
                               & (tokens_df["char_att_agent"] == -1)
                               & (tokens_df["syntactic_head_ID"].isin(verb_rows['token_ID_within_document']))]

    merged_df = pd.merge(conj_verb_rows, verb_rows, left_on='syntactic_head_ID', right_on='token_ID_within_document',
                         how='left')
    mention_head_id_list, verb_agent_head_id_list = merged_df['char_att_agent_y'].tolist(), merged_df[
        'token_ID_within_document_x'].tolist()
    del merged_df
    tokens_df.loc[verb_agent_head_id_list, 'char_att_agent'] = mention_head_id_list

    # fill and ensure the column is of integer type
    tokens_df['char_att_agent'] = tokens_df['char_att_agent'].fillna(-1).astype(int)

    return tokens_df

def patient_attribution(tokens_df):
    tokens_df['char_att_patient'] = -1

    verbs_rows_ids = tokens_df[(tokens_df['POS_tag'] == 'VERB')
    ]["token_ID_within_document"].tolist()

    mention_head_object_row = tokens_df[(tokens_df['dependency_relation'] == 'obj')
                                        & (tokens_df['syntactic_head_ID'].isin(verbs_rows_ids))
                                        & (tokens_df['is_mention_head'] == 1)]

    tokens_df.loc[mention_head_object_row['syntactic_head_ID'].tolist(), 'char_att_patient'] = mention_head_object_row[
        'token_ID_within_document'].tolist()

    # fill and ensure the column is of integer type
    tokens_df['char_att_patient'] = tokens_df['char_att_patient'].fillna(-1).astype(int)

    return tokens_df

def modifiers_attribution(tokens_df):
    tokens_df['char_att_mod'] = -1

    # Create a dictionary mapping token IDs to POS tags
    id_to_pos = dict(zip(tokens_df['token_ID_within_document'], tokens_df['syntactic_head_ID']))
    tokens_df['syntactic_head_head_id'] = tokens_df['syntactic_head_ID'].map(id_to_pos)

    is_mention_head_ids = tokens_df[tokens_df['is_mention_head'] == 1]["token_ID_within_document"].tolist()
    mention_adjectives = tokens_df[(tokens_df['POS_tag'] == "ADJ")
                                   & (tokens_df['dependency_relation'] != "flat:name")
                                   & (tokens_df['syntactic_head_ID'].isin(is_mention_head_ids))
                                   ]
    mention_adjectives_ids = mention_adjectives.index.tolist()
    tokens_df.loc[mention_adjectives_ids, 'char_att_mod'] = tokens_df['syntactic_head_ID']
    # ---
    mention_adjectives = tokens_df[(tokens_df['POS_tag'] == "ADJ")
                                   & (tokens_df['syntactic_head_ID'].isin(mention_adjectives_ids))
                                   ]
    mention_adjectives_ids = mention_adjectives.index.tolist()
    tokens_df.loc[mention_adjectives_ids, 'char_att_mod'] = tokens_df['syntactic_head_head_id']
    # ---
    adj_rows_id = tokens_df[(tokens_df['POS_tag'] == 'ADJ')
                            #                         &(tokens_df['mention_adj_head_id'] == 0)
                            & (tokens_df['dependency_relation'] == 'amod')
                            ]["token_ID_within_document"].tolist()
    mention_head_where_syntactic_head_is_adj = tokens_df[(tokens_df['is_mention_head'] == 1)
                                                         & (tokens_df['dependency_relation'].isin(['advmod']))
                                                         & (tokens_df['syntactic_head_ID'].isin(adj_rows_id))
                                                         ]
    mention_head_id_list, adj_id_list = mention_head_where_syntactic_head_is_adj['token_ID_within_document'].tolist(), \
    mention_head_where_syntactic_head_is_adj['syntactic_head_ID'].tolist()
    tokens_df.loc[adj_id_list, 'char_att_mod'] = mention_head_id_list

    # --- être
    être_head_ids = tokens_df[(tokens_df['lemma'] == 'être') & (tokens_df['POS_tag'] == 'AUX')][
        'syntactic_head_ID'].tolist()
    mention_head_ids = tokens_df[(tokens_df['is_mention_head'] == 1)
                                 & (tokens_df['dependency_relation'] == 'nsubj')
                                 ]
    adj_rows_id = tokens_df[(tokens_df['POS_tag'].isin(['ADJ', 'NOUN']))
                            #                         & (tokens_df['is_mention_head'] == 0)
                            & (tokens_df['token_ID_within_document'].isin(être_head_ids))
                            & (tokens_df['token_ID_within_document'].isin(
        mention_head_ids['syntactic_head_ID'].tolist()))
                            ].index.tolist()

    mention_head_with_adj = tokens_df[
        (tokens_df['token_ID_within_document'].isin(mention_head_ids['token_ID_within_document'].tolist()))
        & (tokens_df['syntactic_head_ID'].isin(adj_rows_id))]
    mention_head_id_list, adj_id_list = mention_head_with_adj['token_ID_within_document'].tolist(), \
    mention_head_with_adj['syntactic_head_ID'].tolist()
    tokens_df.loc[adj_id_list, 'char_att_mod'] = mention_head_id_list
    # --- conj other adjective

    mention_adj_ids = tokens_df[(tokens_df['char_att_mod'] != -1)]['token_ID_within_document'].tolist()
    conj_adj_rows = tokens_df[(tokens_df['POS_tag'] == 'ADJ')
                              & (tokens_df['char_att_mod'] == -1)
                              & (tokens_df['syntactic_head_ID'].isin(mention_adj_ids))
                              ]

    merged_df = pd.merge(conj_adj_rows, tokens_df, left_on='syntactic_head_ID', right_on='token_ID_within_document',
                         how='left')
    conj_adj_id_list, mention_head_id_list = merged_df['token_ID_within_document_x'].tolist(), merged_df[
        'char_att_mod_y'].tolist()
    del merged_df

    tokens_df.loc[conj_adj_id_list, 'char_att_mod'] = mention_head_id_list

    # fill and ensure the column is of integer type
    tokens_df['char_att_mod'] = tokens_df['char_att_mod'].fillna(-1).astype(int)

    return tokens_df

def avoir_poss_attribution(tokens_df):
    tokens_df['avoir_object_mention_head_id'] = -1

    avoir_ids = tokens_df[(tokens_df["lemma"] == 'avoir') & (tokens_df["POS_tag"] == 'VERB')][
        "token_ID_within_document"].tolist()
    avoir_mention_head_subject_ids = tokens_df[(tokens_df['dependency_relation'] == "nsubj")
                                               & (tokens_df['syntactic_head_ID'].isin(avoir_ids))
                                               & (tokens_df['is_mention_head'] == 1)
                                               ]
    avoir_with_object_ids = tokens_df[(tokens_df['dependency_relation'] == "obj")
                                      & (tokens_df['POS_tag'].isin(['NOUN', 'ADJ']))
                                      & (tokens_df['syntactic_head_ID'].isin(avoir_ids))
                                      ]
    merged_df = pd.merge(avoir_with_object_ids, avoir_mention_head_subject_ids, left_on='syntactic_head_ID',
                         right_on='syntactic_head_ID', how='left')
    merged_df = merged_df[~merged_df["token_ID_within_document_y"].isna()]
    merged_df['token_ID_within_document_y'] = merged_df['token_ID_within_document_y'].astype(int)

    avoir_object_id_list, mention_head_id_list = merged_df['token_ID_within_document_x'].tolist(), merged_df[
        'token_ID_within_document_y'].tolist()
    del merged_df

    tokens_df.loc[avoir_object_id_list, 'avoir_object_mention_head_id'] = mention_head_id_list

    tokens_df['avoir_object_mention_head_id'] = tokens_df['avoir_object_mention_head_id'].fillna(-1)
    tokens_df['avoir_object_mention_head_id'] = tokens_df['avoir_object_mention_head_id'].astype(int)
    tokens_df[tokens_df["avoir_object_mention_head_id"] != -1]
    return tokens_df
def articles_poss_attribution(tokens_df):
    possessive_articles_list = ['mon', 'ton', 'son', 'ma', 'ta', 'sa', 'notre', 'votre', 'leur', 'mes', 'mes', 'tes',
                                'ses', 'nos', 'vos', 'leurs']
    tokens_df['possesive_head_id'] = -1

    possessive_articles_mention_head = tokens_df[(tokens_df["word"].str.lower().isin(possessive_articles_list))
                                                 & (tokens_df["is_mention_head"] == 1)
                                                 & (tokens_df["POS_tag"] == ('DET'))
                                                 ]

    possessive_article_object = tokens_df[
        (tokens_df["token_ID_within_document"].isin(possessive_articles_mention_head['syntactic_head_ID'].tolist()))
        & (tokens_df["POS_tag"].isin(['NOUN', 'PROPN']))
        ]

    merged_df = pd.merge(possessive_article_object, possessive_articles_mention_head,
                         left_on='token_ID_within_document', right_on='syntactic_head_ID', how='left')
    merged_df[['sentence_ID_x', 'token_ID_within_document_x', 'token_ID_within_document_y']]

    mention_head_id_list, poss_object_id_list = merged_df['token_ID_within_document_y'].tolist(), merged_df[
        'token_ID_within_document_x'].tolist()
    del merged_df

    tokens_df.loc[poss_object_id_list, 'possesive_head_id'] = mention_head_id_list
    tokens_df['possesive_head_id'] = tokens_df['possesive_head_id'].fillna(-1)
    tokens_df['possesive_head_id'] = tokens_df['possesive_head_id'].astype(int)

    len(tokens_df[tokens_df['possesive_head_id'] != -1])
    tokens_df[tokens_df['possesive_head_id'] != -1]

    return tokens_df
def nmod_poss_attribution(tokens_df):
    tokens_df['nmod_possesive_head_id'] = -1
    de_rows = tokens_df[(tokens_df['lemma'] == 'de')]
    mention_head_nmod_rows = tokens_df[(tokens_df['is_mention_head'] == 1)
                                       & (tokens_df['dependency_relation'] == 'nmod')
                                       & (tokens_df['token_ID_within_document'].isin(
        de_rows['syntactic_head_ID'].tolist()))]
    possessiv_nominal_modifier = tokens_df[(tokens_df['POS_tag'] == 'NOUN')
                                           & (tokens_df['token_ID_within_document'].isin(
        mention_head_nmod_rows['syntactic_head_ID'].tolist()))]

    merged_df = pd.merge(possessiv_nominal_modifier, mention_head_nmod_rows, left_on='token_ID_within_document',
                         right_on='syntactic_head_ID', how='left')
    merged_df[['sentence_ID_x', 'token_ID_within_document_x', 'token_ID_within_document_y']]

    mention_head_id_list, poss_object_id_list = merged_df['token_ID_within_document_y'].tolist(), merged_df[
        'token_ID_within_document_x'].tolist()
    del merged_df

    tokens_df.loc[poss_object_id_list, 'nmod_possesive_head_id'] = mention_head_id_list
    tokens_df['nmod_possesive_head_id'] = tokens_df['nmod_possesive_head_id'].fillna(-1)
    tokens_df['nmod_possesive_head_id'] = tokens_df['nmod_possesive_head_id'].astype(int)

    len(tokens_df[tokens_df['nmod_possesive_head_id'] != -1])
    tokens_df[tokens_df['nmod_possesive_head_id'] != -1]

    return tokens_df
def combine_poss_columns(tokens_df):
    # combine poss columns
    # Create the new column using np.where
    tokens_df['char_att_poss'] = np.where(
        tokens_df['possesive_head_id'] != -1,  # If possesive_head_id is not 0
        tokens_df['possesive_head_id'],  # Use possesive_head_id
        np.where(
            tokens_df['avoir_object_mention_head_id'] != -1,  # Else, if avoir_object_mention_head_id is not 0
            tokens_df['avoir_object_mention_head_id'],  # Use avoir_object_mention_head_id
            -1  # Otherwise, use 0
        )
    )

    tokens_df['char_att_poss'] = np.where(
        tokens_df['char_att_poss'] != -1,  # If possesive_head_id is not 0
        tokens_df['char_att_poss'],  # Use possesive_head_id
        np.where(
            tokens_df['nmod_possesive_head_id'] != -1,  # Else, if avoir_object_mention_head_id is not 0
            tokens_df['nmod_possesive_head_id'],  # Use avoir_object_mention_head_id
            -1  # Otherwise, use 0
        )
    )
    tokens_df = tokens_df.drop(columns=['avoir_object_mention_head_id', 'possesive_head_id', 'nmod_possesive_head_id'])
    return tokens_df

def possessive_attribution(tokens_df):
    tokens_df = avoir_poss_attribution(tokens_df)
    tokens_df = articles_poss_attribution(tokens_df)
    tokens_df = nmod_poss_attribution(tokens_df)

    tokens_df = combine_poss_columns(tokens_df)

    if 'syntactic_head_head_id' in tokens_df.columns:
        tokens_df = tokens_df.drop(columns=['syntactic_head_head_id'])

    return tokens_df

def extract_attributes(entities_df, tokens_df):
    head_tokens_ids = entities_df['head_id'].tolist()
    tokens_df['is_mention_head'] = 0
    tokens_df.loc[head_tokens_ids, 'is_mention_head'] = 1

    tokens_df = agent_attribution(tokens_df)
    tokens_df = patient_attribution(tokens_df)
    tokens_df = modifiers_attribution(tokens_df)
    tokens_df = possessive_attribution(tokens_df)

    return tokens_df