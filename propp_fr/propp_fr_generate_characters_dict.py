from collections import Counter

def gender_inference(gender_list):
    mention_count = len(gender_list)
    gender_count = Counter(gender_list)
    male_count, female_count = gender_count['Male'], gender_count['Female']
    gendered_count = male_count + female_count
    if gendered_count == 0:
        gendered_ratio, male_ratio, female_ratio = 0, 0, 0
    else:
        gendered_ratio = gendered_count / mention_count
        male_ratio, female_ratio = male_count / gendered_count, female_count / gendered_count

    # argmax
    argmax_threshold = 0.015
    if (male_ratio == female_ratio) or (gendered_ratio < argmax_threshold):
        argmax = 'Null'
    else:
        if male_ratio > female_ratio:
            argmax = 'Male'
        else:
            argmax = 'Female'

    char_gender_dict = {'ratio': round(gendered_ratio, 4),
                        'inference': {'Male': round(male_ratio, 2), 'Female': round(female_ratio, 2)},
                        'max': round(max([male_ratio, female_ratio]), 2),
                        'argmax': argmax,
                        }
    return char_gender_dict

def number_inference(number_list):
    mention_count = len(number_list)
    number_count = Counter(number_list)
    singular_count, plural_count = number_count['Singular'], number_count['Plural']
    numbered_count = singular_count + plural_count
    if numbered_count == 0:
        numbered_ratio, singular_ratio, plural_ratio = 0, 0, 0
    else:
        numbered_ratio = numbered_count / mention_count
        singular_ratio, plural_ratio = singular_count / numbered_count, plural_count / numbered_count

    # argmax
    argmax_threshold = 0.015
    if (singular_ratio == plural_ratio) or (numbered_count < argmax_threshold):
        argmax = 'Null'
    else:
        if singular_ratio > plural_ratio:
            argmax = 'Singular'
        else:
            argmax = 'Plural'

    char_number_dict = {'ratio': round(numbered_ratio, 4),
                        'inference': {'Singular': round(singular_ratio, 2), 'Plural': round(plural_ratio, 2)},
                        'max': round(max([singular_ratio, plural_ratio]), 2),
                        'argmax': argmax,
                        }
    return char_number_dict

def get_mentions(COREF_group):
    mentions_dict = {}
    for mention_type in ['PROP', 'NOM', 'PRON']:
        mentions_dict[mention_type] = []
        mentions_list = COREF_group[COREF_group['prop'] == mention_type]['text'].tolist()
        if not mention_type == 'PROP':
            mentions_list = [mention.lower() for mention in mentions_list]
        mention_counter = Counter(mentions_list).most_common()
        for mention in mention_counter:
            mentions_dict[mention_type].append({'n': mention[0], 'c': mention[1]})

    # Mapping of old keys to new keys
    key_mapping = {'PROP': 'proper', 'NOM': 'common', 'PRON': 'pronoun'}
    # Create a new dictionary with the updated keys
    mentions_dict = {key_mapping[old_key]: value for old_key, value in mentions_dict.items()}

    return mentions_dict

def extract_char_attributs(COREF_group, tokens_df, attributes_column):
    # Use boolean indexing for filtering
    filtered_tokens_df = tokens_df[tokens_df[attributes_column].isin(COREF_group['head_id'])]

    # Use itertuples for creating the list of dictionaries
    char_attributes = [{'w': row.lemma, 'i': row.token_ID_within_document} for row in
                       filtered_tokens_df.itertuples(index=False)]

    return char_attributes

def generate_characters_dict(tokens_df,
                            entities_df,
                            COREF_column='COREF'):
    tokens_df['lemma'] = tokens_df.copy()['lemma'].str.lower()
    tokens_df = tokens_df[
        ['token_ID_within_document', 'word', 'lemma', 'char_att_poss', 'char_att_agent', 'char_att_patient',
         'char_att_mod']]

    PER_entities_df = entities_df[entities_df['cat'] == 'PER']
    PER_entities_df = PER_entities_df.sort_values(by=[COREF_column, 'start_token'])
    characters_book_file = {'characters': []}

    # Iterate over groups of rows grouped by 'COREF'
    for COREF, COREF_group in PER_entities_df.groupby(COREF_column):
        char_id = COREF
        char_count = len(COREF_group)
        char_mention_ratio = round(char_count / len(PER_entities_df), 4)
        if char_count >= 2:
            char_gender = gender_inference(COREF_group['gender'].tolist())
            char_number = number_inference(COREF_group['number'].tolist())
            char_mentions = get_mentions(COREF_group)
            char_poss = extract_char_attributs(COREF_group, tokens_df, 'char_att_poss')
            char_agent = extract_char_attributs(COREF_group, tokens_df, 'char_att_agent')
            char_patient = extract_char_attributs(COREF_group, tokens_df, 'char_att_patient')
            char_mod = extract_char_attributs(COREF_group, tokens_df, 'char_att_mod')

            new_character = {'id': char_id,
                             'count': {'occurrence': char_count, 'mention_ratio': char_mention_ratio},
                             'gender': char_gender,
                             'number': char_number,
                             'mentions': char_mentions,
                             'agent': char_agent,
                             'patient': char_patient,
                             'mod': char_mod,
                             'poss': char_poss,
                             }
            characters_book_file['characters'].append(new_character)

    return characters_book_file