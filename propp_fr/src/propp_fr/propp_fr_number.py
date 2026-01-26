#%%
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm
import json
import os
import csv
from time import time # To time our operations
tqdm.pandas()
#%%
def clean_column(df, column):
    df[column] = df[column].str.lower()
    df[column] = df[column].fillna('')
    # Replacement operations
    df[column] = df[column].replace(r'\!', '', regex=True)
    df[column] = df[column].replace(r'[–—―‒]', '-', regex=True)  # Replace different dash characters with hyphen
    df[column] = df[column].replace(r'\.\.\.', ' ', regex=True)
    df[column] = df[column].replace(r'\*', ' ', regex=True)
    df[column] = df[column].replace(r'"', '', regex=True)
    df[column] = df[column].replace(r'‶', '', regex=True)
    df[column] = df[column].replace(r'«', '', regex=True)
    df[column] = df[column].replace(r'\.\-', '', regex=True)  # Remove all occurrences of '.-'
    df[column] = df[column].replace(r'^[-.]', '', regex=True)  # Remove leading '-' or '.'
    df[column] = df[column].replace(r'�', '', regex=True)
    df[column] = df[column].replace(r'’', "'", regex=True)
    df[column] = df[column].replace(r'\[', " ", regex=True)
    df[column] = df[column].replace(r'\]', " ", regex=True)
    df[column] = df[column].replace(r'\(', " ", regex=True)
    df[column] = df[column].replace(r'\)', " ", regex=True)
    df[column] = df[column].replace(r'[ \t]{2,}', " ", regex=True)  # Collapse multiple spaces/tabs to a single space
    df[column] = df[column].str.strip()  # Remove leading and trailing spaces
    return df
#%%
def exact_ngram_match(df):
    filtered_df = df[df['number'] == 'Not_Assigned']
    
    singular_ngram_list = ['il', '-il', 'je', '-je', 'elle', '-elle','son', 'lui', 'sa', 'ses', 'j’', "j'", 'mon', 'tu', 'me', 'moi', 'ma', 'm’', "m'", 'le', 'l’', "l'" ,'la', 'mes', 'tes', '-il', 'te', 'toi', 'monsieur', 't’', 'ton', 'madame', 'ta', '-moi', 'maman', 'mademoiselle', 'celui-ci', 'celle-ci', "t'", 'lui-même', 'monseigneur', 'ce', 'celui', 'celle', 'cette', 'cet', 'papa', 'lequel', 'laquelle', 'quelqu’ un', 'elle-même', 'père', 'mère', 'sire', 'mienne', 'mien', 'dame', 'docteur', 'capitaine', 'seigneur', 'sien', 'homme', 'sienne', 'patron', 'femme', 'j', 'général', 'chérie', 'une seule et même personne', 'son seigneur et maître', 'une femme jeune et belle', 'auquel', "quelqu' un", 'chevalier', 'grand-père', 'commissaire', 'reine', 'prince', 'enfant', 'frère', 'petite', 'grand-mère', 'ami', 'comte', 'chéri', 'garçon', 'commandant', 'messire', 'prêtre', 'petit', 'miss', 'duc', 'chef', 'citoyen', 'comtesse', 'princesse', 'baron', 'colonel', 'mec', 'bijou', 'camarade', 'cousin', 'lieutenant', 'misérable', 'mignonne', 'maître', 'maîtresse', 'nana', 'un', 'une', 'notre-dame', 'fillette', 'duquel', 'bonhomme', 'médecin', 'cher', 'sir', 'chère', 'quelle', 'brigadier', 'parrain', 'sergent', "grand'mère", 'compère', 'professeur', 'altesse', 'vicomte', 'roi', 'berger', 'duchesse', 'cadet', 'moine', 'maréchal', 'inspecteur', 'tante', 'juge', 'tonton', 'nourrice', 'excellence', 'cousine', 'citoyenne', 'curé', 'imbécile', 'bonne-maman', 'voleur', 'tienne', 'tien', 'nounou', 'major', 'cardinal', 'mari', 'gamin', 'enfant', 'marraine', 'fiston', 'magistrat', 'soldat', 'demoiselle', 'vieillard', 'nénette', 'm’ sieur', 'monte - cristo', 'myrtille', 'reinette', 'coco', 'forestier', 'chouchou', 'marin', 'compagnon', 'brigand', 'petite-reine', 'coquin', 'assassin', 'baronne', 'gouverneur', 'canaille', 'flic', 'président', 'mémé', 'député', 'chien', 'bourreau', 'oncle', 'malheureuse', 'malin', 'amie', 'ministre', 'voisin', 'cavalier', 'mamzelle', 'saint', 'caporal', 'animal', 'fille', 'doc', 'évêque', 'bébé', 'sœur', 'neveu', 'môme', 'officier', 'trésor', 'gendre', 'avocat', 'cocher', 'démon', 'pauvre', 'mouton', 'paysan', 'maire', 'bandit', 'étranger', 'connétable', 'bouledogue', 'musicien', 'beau-père', 'grand-papa', 'gendarme', 'peste', 'parisien', 'malade', 'fidèle', 'guide', '-tu', "l'enfant", "quelqu'un", 'mouton', "celui-là", 'celle-là', 'brigand', "-j'", "- la", 'toute seule'
                          ]
    plural_ngram_list = ['nous', 'ils', 'elles', 'eux', 'leur', 'leurs', 'les', 'notre', 'nos', 'tous', 'toutes', 'ceux', 'lesquels', 'lesquelles', 'auxquels', 'auxquelles', 'hommes', 'femmes', 'mesdames', 'enfants', 'messieurs',
                        '-ils', '-nous', 'amis', 'amies', 'nôtre', 'deux', 'soldats', 'mesdemoiselles', 'citoyens', 'camarades', 'frères', 'beaucoup', 'brigands', 'seigneurs', 'prêtres', 'quelques-uns', 'celles', 'messeigneurs', 'garçons',
                         'desquels', 'nôtres', 'plusieurs', 'voyageurs', 'dames', 'parents', 'certains', 'moines', 'bandits', 'trois', 'matelots', 'treize', 'fous', 'voleurs', 'musiciens', 'marins', 'compagnons', 'cousins',
                         'maîtres', 'juges', 'médecins', 'princes', 'ministres', 'chevaliers', 'vieillards', 'courtisans', 'messires', 'valets', 'd’ autres', 'vous autres', 'vous deux', 'nous deux', 'eux deux'
                        ]
    ambiguous_ngram_list = ['vous', '-vous', 'votre', 'vôtre', 'vos', 'qui', 'que', "qu'", 'qu’','on', 'dont', "c'", 'c’', 'personne', 'chacun', 'chaque', 'en', 's’', "s'", 'bourgeois', 'malheureux', 'fils', 'gars', 'vieux', 'ça',
                            'français', 'aucun', 'aucune']

    singular_rows = filtered_df[(filtered_df['text'].isin(singular_ngram_list))]
    df.loc[singular_rows.index, 'number'] = 'Singular'
    plural_rows = filtered_df[(filtered_df['text'].isin(plural_ngram_list))]
    df.loc[plural_rows.index, 'number'] = 'Plural'
    ambiguous_rows = filtered_df[(filtered_df['text'].isin(ambiguous_ngram_list))]
    df.loc[ambiguous_rows.index, 'number'] = 'Ambiguous'
    
    del filtered_df
    return df

def et_ou_mentions(df):
    filtered_df = df[df['number'] == 'Not_Assigned']
    
    pattern = '|'.join([' et ', ' ou ', ' ni ', ' et à '])
    plural_rows = filtered_df[(filtered_df['text'].str.contains(pattern))
                              # &(filtered_df['mention_len'] <= 6)
                              &(filtered_df['in_to_out_nested_level'] > 0)
                              &(filtered_df['nested_entities_count'] >= 1)]
    df.loc[plural_rows.index, 'number'] = 'Plural'
    
    plural_rows = filtered_df[(filtered_df['text'].str.contains(pattern))
                          &(filtered_df['mention_len'] == 3)
                          &(filtered_df['in_to_out_nested_level'] > 0)]
    df.loc[plural_rows.index, 'number'] = 'Plural'

    del filtered_df
    return df

def les_mentions(df):
    filtered_df = df[df['number'] == 'Not_Assigned']

    les_rows = filtered_df[(filtered_df['text'].str.startswith(tuple(['les ', 'ces '])))]
    df.loc[les_rows.index, 'number'] = 'Plural'

    del filtered_df
    return df

def grouped_individuals(df):
    filtered_df = df[df['number'] == 'Not_Assigned']
    
    grouped_individuals_head = ['bourgeoisie', 'masse', 'tas', 'maison', 'chœur', 'peuple', 'troupeau', 'tablée', 'équipage', 'assistance', 'plupart', 'multitude', 'personnel', 'entourage', 'populace', 'orchestre', 'escorte', 'couvée', 'jeunesse', 'compagnie', 'patrouille', 'bande', 'assemblée', 'foule', 'groupe', 'famille', 'belle-famille', 'monde', 'population', 'troupe', 'noblesse', 'village', 'gens', 'tribu', 'nation', 'ville', 'société', 'colonie', 'pays', 'cité', 'régiment', 'bataillon', 'escadron', 'royaume', 'armée', 'milice', 'cohorte', 'caravane', 'horde', 'garnison', 'division', 'contrée', 'province', 'capitale', 'quartier', 'secteur', 'ordre', 'confrérie', 'communauté', 'cour', 'population', 'couple', 'police', 'ménage', 'brigade', 'public', 'gouvernement', 'auditoire', 'cortège', 'section', 'camp', 'clergé', 'paroisse', 'cavalerie', 'république', 'département', 'commune', 'corporation', 'bourg', 'comté', 'monastère', 'couvent', 'diocèse', 'ministère', 'bateau', 'conseil', 'nombreux', 'nombreuses', 'clientèle', 'personnes', 'vamile']
    plural_head = ['amis', 'diables', 'soldats', 'filles', 'femmes', 'enfants', 'hommes', 'arrivants', 'garçons', 'dames', 'connaissances', 'personnes', 'personnages', 'camarades', 'âmes', 'domestiques', 'seigneurs', 'serviteurs', 'recrues', 'parents', 'paysans', 'hôtes', 'mortels', 'messieurs', 'rentiers', 'officiers', 'coupables', 'couples', 'policiers', 'innocents', 'employés', 'chrétiens', 'témoins', 'visiteurs', 'passants', 'troupes', 'mariés', 'promeneurs', 'drôles', 'invités', 'ouvriers', 'groupes', 'êtres', 'gardes', 'marchands', 'misérables', 'spectateurs', 'combattants', 'équipages', 'gendarmes', 'journalistes', 'mères', 'autres', 'habitants', 'messieurs', 'messieurs', 'mesdames', 'mlles', 'mmes', 'mm.', 'mesdemoiselles']
    grouped_individuals_rows = filtered_df[(filtered_df['head_word'].isin(grouped_individuals_head+plural_head))]
    df.loc[grouped_individuals_rows.index, 'number'] = 'Plural'
    del filtered_df
    return df

def proper_mention(df):
    filtered_df = df[df['number'] == 'Not_Assigned']
    filtered_df = filtered_df[filtered_df['prop'] == 'PROP']
    
    ambiguous_proper_mentions = filtered_df[(filtered_df['text'].str.startswith(tuple(['des '])))
                                      # &(filtered_df['mention_len'] <= 4)
                                        ]
    df.loc[ambiguous_proper_mentions.index, 'number'] = 'Ambiguous'
    
    singular_proper_mention_rows = filtered_df[~(filtered_df['text'].str.startswith(tuple(['des '])))
                                                 &(filtered_df['number'] == 'Not_Assigned')
                                               # &(filtered_df['mention_len'] <= 4)
                                        ]
    df.loc[singular_proper_mention_rows.index, 'number'] = 'Singular'
    del filtered_df
    return df

def prefix_match(df):
    filtered_df = df[df['number'] == 'Not_Assigned']
    
    singular_prefix_list = [prefix+' ' for prefix in ['au', 'baron', 'beau', 'bel', 'belle', 'bon', 'bonne', 'ce', 'celle', 'celui', 'cet', 'cette', 'cher', 'chère', 'comte', 'cousin', 'cousine', 'dame', 'docteur', 'dom', 'don', 'donna', 'don', 'dom'
                                                      'dr', 'du', 'elle', 'feu', 'jeune', "l'", "l' un", "l' une", 'la', 'lady', 'le', 'leur', 'lord', 'lui', 'l’', 'l’ un', 'l’ une', 'm.', 'ma', 'madame', 'mademoiselle',
                                                      'maman', 'marquise', 'maître', 'me', 'messire', 'mgr', 'milord', 'miss', 'mister', 'mistress', 'mlle', 'mlle.', 'mme', 'mme.', 'moi', 'mon', 'monseigneur', 'monsieur', 'mr',
                                                      'mr.', 'mrs', 'mère', 'notre', 'papa', 'pauvre', 'petit', 'petite', 'père', 'grande', 'quel', 'quelle', 'sa', 'saint', 'sainte', 'seigneur', 'sir', 'sir.', 'soi', 'son', 'sœur', 'ta',
                                                      'tante', 'toi', 'ton', 'un', 'une', 'votre', 'tonton', 'brave', 'malheureuse', 'oncle', 'grand', 'honnête']] + ["l’", "l'"]
    plural_prefix_list = [prefix+' ' for prefix in ['autres', 'aux', 'beaucoup', 'celles', 'cent', 'certaines', 'certains', 'ces', 'ceux', 'chers', 'chères', 'bonnes', 'cinq', 'cinquante', 'dames', 'des', 'deux', 'dix', 'dons', 'douze',
                                                    'elles', 'eux', 'femmes', 'feux', 'hommes', 'huit', 'ils', 'les', 'les deux', 'leurs', 'mes', 'mesdames', 'messieurs', 'messires', 'messrs', 'mgrs', 'mille', 'misters',
                                                    'mistresses', 'mm.', 'mm .', 'mmes', 'neuf', 'nombreux', 'nos', 'nous', 'onze', 'petits', 'plusieurs', 'quarante', 'quatre', 'quelques', 'quels', 'saintes', 'saints', 'sept', 'ses',
                                                    'six', 'soixante', 'tes', 'tous', 'tout', 'toute', 'toutes', 'trente', 'trois', 'vingt', 'vos', 'quatorze']]
    ambiguous_prefix_list = [prefix+' ' for prefix in ['vous', 'aucun', 'aucune']]
    
    filtered_df = filtered_df[(filtered_df['mention_len'] > 1) & (filtered_df['mention_len'] <= 4)]
    singular_rows = filtered_df[filtered_df['text'].str.startswith(tuple(singular_prefix_list))]
    df.loc[singular_rows.index, 'number'] = 'Singular'
    plural_rows = filtered_df[filtered_df['text'].str.startswith(tuple(plural_prefix_list))]
    df.loc[plural_rows.index, 'number'] = 'Plural'
    ambiguous_rows = filtered_df[filtered_df['text'].str.startswith(tuple(ambiguous_prefix_list))]
    df.loc[ambiguous_rows.index, 'number'] = 'Ambiguous'
   
    del filtered_df
    return df
#%%
def assign_number_to_PER_entities(PER_entities_df):
    PER_entities_df['number'] = 'Not_Assigned'
    PER_entities_df = clean_column(PER_entities_df, 'text')
    PER_entities_df = clean_column(PER_entities_df, 'head_word')
    
    PER_entities_df = exact_ngram_match(PER_entities_df)
    PER_entities_df = et_ou_mentions(PER_entities_df)
    PER_entities_df = les_mentions(PER_entities_df)
    PER_entities_df = grouped_individuals(PER_entities_df)
    PER_entities_df = proper_mention(PER_entities_df)
    PER_entities_df = prefix_match(PER_entities_df)
    return PER_entities_df
