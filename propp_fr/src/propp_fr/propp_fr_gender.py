#%%
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm
import json
import os
import csv
from time import time # To time our operations
tqdm.pandas()
import re  # Import the re module
import numpy as np
import plotly.express as px
#%%
def clean_column(df, column):
    df[column] = df[column].str.lower()
    df[column] = df[column].fillna('')
    # Replacement operations
    df[column] = df[column].replace(r'\!', '', regex=True)
    df[column] = df[column].replace(r'[–—―‒]', '-', regex=True)  # Replace different dash characters with hyphen
    df[column] = df[column].replace(r'\.\.\.', ' ', regex=True)
    df[column] = df[column].replace(r'\*', ' ', regex=True)
    df[column] = df[column].replace(r'\/', ' ', regex=True)
    df[column] = df[column].replace(r'\°', ' ', regex=True)
    df[column] = df[column].replace(r'"', '', regex=True)
    df[column] = df[column].replace(r'‶', '', regex=True)
    df[column] = df[column].replace(r'«', '', regex=True)
    df[column] = df[column].replace(r'\.\-', '', regex=True)  # Remove all occurrences of '.-'
    df[column] = df[column].replace(r"^[-.',]", '', regex=True)  # Remove leading '-' or '.'
    df[column] = df[column].replace(r"^[-.',]", '', regex=True)  # Remove leading '-' or '.'
    df[column] = df[column].replace(r"^[-.',]", '', regex=True)  # Remove leading '-' or '.'
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

#%%
def exact_ngram_match(df):
    filtered_df = df[df['gender'] == 'Not_Assigned']

    male_ngram_list = ['il', '-il', 'messieurs', 'mossieur', 'lui-même', 'le', 'celui-ci', 'le même', 'monsieur', 'grand-père', 'papa', 'celui', 'prêtre', 'homme', 'père', 'auquel', 'lequel', '-le', 'frère', 'm.', 'prêtres', 'garçon', 'un', 'mari', 'magicien', 'manants', 'matelots', 'monde', 'lesquels', 'praticien', 'pape', 'paysan', 'salop', 'sire', 'maître', 'seigneur', 'patron', 'général', 'prince', 'chevalier', 'commissaire', 'comte', 'chéri', 'commandant', 'augustin', 'ami', 'messire', 'duc', 'citoyen', 'petit', 'romain', 'baron', 'chef', 'mec', 'colonel', 'duquel', 'lieutenant', 'cousin', 'roi', 'bonhomme', 'jonas', 'cher', 'professeur', 'parrain', 'vicomte', 'berger', 'maréchal', 'brigadier', 'cadet', 'tonton', 'voleur', 'gendre', 'page', 'gamin', 'sir', 'assassin', 'cardinal', 'major', 'soldat', 'moine', 'fiston', 'curé', 'vieillard', 'brigand', 'coquin', 'cavalier', 'marin', 'gouverneur', 'malin', 'inspecteur', 'gringalet', 'messer', 'monsir', 'musicien', 'mort', 'voisin', 'président', 'évêque', 'oncle', 'député', 'cocher', 'aimé', 'compagnon', 'caporal', 'neveu', 'avocat', 'officier', 'connétable', 'signor', 'canonnier', 'pélerin', 'beau-père', 'parisien', 'étranger', 'prévot', 'malais', 'magistrat', 'lutin', 'grand-papa', 'amant', 'comtois', 'musard', 'barbier', 'marchand', 'sculpteur', 'négociant', 'croyant', 'midshipman', 'viennois', 'fils', 'frères', 'gars', 'garçons', 'seigneurs', 'guerrier', 'princes', 'rois', 'moines', 'sultan', 'pasteur', "l' un", 'celui - ci', 'celui -là', 'lui - même', 'les deux hommes', 'deux hommes', 'ces deux hommes', 'les trois hommes', 'trois hommes', 'des deux hommes', 'quatre hommes', 'plusieurs hommes', 'les quatre hommes', "-l' un", 'mon cher', 'du même', 'moi tout seul', 'le christ', 'lui aussi', 'mon père', 'son père']

    female_ngram_list = ['elle', 'elles', '-elle', 'elle-même', 'la', 'celle-ci', 'celle', 'la même', 'madame', 'maman', 'laquelle', 'femme', 'mademoiselle', 'femmes', 'toutes', "grand'mère", 'est-elle', 'chacune', 'auxquelles', 'filles', 'mesdames', 'quelle', 'mère', 'grand-mère', 'mesdemoiselles', 'vierges', 'épouses', 'reine', 'mères', 'salope', 'dame', 'chérie', 'nana', 'angélique', 'miss', 'princesse', 'manette', 'une', 'comtesse', 'maîtresse', 'fillette', 'chère', 'mignonne', 'reinette', 'duchesse', 'nounou', 'cocotte', 'citoyenne', 'cousine', 'tante', 'nourrice', 'marraine', 'demoiselle', 'cette', 'baronne', 'bonne-maman', 'amie', 'mémé', 'fille', 'chrétienne', 'sœur', 'mamzelle', 'malheureuse', 'madone', 'bohémienne', 'bergère', 'mamselle', 'friponne', 'égyptienne', 'aïeule', 'celles', 'lesquelles', 'dames', 'mamie', 'ele', "l' une", 'celle - ci', 'celle -là', 'elle - même', 'nous toutes', "la sienne", 'moi toute seule', 'la tienne', 'la bonne', 'elle aussi', 'la mienne', 'toute seule'
                        ]
    ambiguous_ngram_list = ['je', 'vous', 'ma', 'ta', 'sa', 'son', 'mon', 'ton', 'qui', 'dont', 'les', 'me', '-vous', '-je', 'ce', '-ce', '-nous', 'nous', 'ses', 'moi', 'moi-même', 'vous-même', 'tu', 'te', 'toi', 'toi-même', 'lui', "j'", "l'", 'ils', "m'", 'leur', 'on', 'mes', 'où', 'les', 'notre', 'leurs', "c'", 'votre', 'eux', 'nos', "t'", '-moi', "qu'", 'que', 'en', 'tous', 'vos', 'se', 'chacun', "s'", 'tes', '-ils', 'mienne', 'tienne', 'ceux', 'enfant', 'y', 'soi', 'sien', '-on', 'ça', 'docteur', 'gens', 'enfants', 'petite', '-toi', 'capitaine', 'ensemble', 'est-on', 'famille', 'philosophe', 'musiciens', 'miennes', 'miens', 'fit', 'fous', 'fut', '-lui', 'vôtre', 'témoins', 'amis', 'couple', 'mien', 'misérable', 'maire', 'violon', 'virtuose', 'violoniste', 'vôtres', 'écuyer', 'été', 'était', 'équipages', 'turcs', 'oiseau', 'oiseaux', 'poète', 'politiques', 'peuple', 'philosophes', 'plusieurs', 'quatre-vingt-dix-neuf', 'danseurs', 'siens', 'sentinelle', 'serait', 'personne', 'hommes', 'j', 'aubert', 'excellence', 'ange', 'camarade', 'tien', 'aux', 'imbécile', 'du', 'médecin', 'chien', 'bébé', 'juge', 'chouchou', 'pauvre', 'de', 'huguet', 'ile', 'vingt', 'casse-cou', 'tours', 'pompon', 'espagnol', 'allemand', 'corsaire', '-tu', 'canaille', 'bourreau', 'gui', 'li', 'ministre', 'peste', 'caro', 'animal', 'môme', 'flic', 'démon', 'moitié', 'vo', 'z', 'baba', 'bichon', 'gendarme', 'morbleu', 'guide', 'mé',  'jemmy', 'malade',  'majesté', 'politique', 'ermite', 'auxquels', 'i', 'autres', 'marquis', 'honoré', 'you', 'deux', 'certains', 'mike', 'treize', 'trois', 'paysans', 'époux', 'parents', 'spectateurs', 'personnes', "-j'", 'étudiants', 'voyageurs', 'voleurs', 'groupe', 'serviteurs', 'gendarmes', 'nôtres', "quelqu' un", 'moi - même', "d' autres", 'toi - même', 'vous - même', 'eux - mêmes', 'tous deux', 'tous les deux', 'les uns', 'ceux - ci', 'voyageurs', 'morts', 'amis', 'domestiques', 'nous deux', 'tous trois', 'nous - mêmes', 'nous autres', 'nous tous', 'nous trois', 'nous - même', 'nous quatre', 'nous cinq', 'nous seuls', 'nous mêmes', 'nous tout seuls', 'nous aussi', 'nous deux seuls', 'nous six', 'nous tous ensemble', 'nous autres poètes', 'toutes deux', 'toutes les deux', 'toutes trois', 'toutes les trois', 'toutes les quatre', 'toutes quatre', 'toutes ensemble', 'toutes les mêmes', 'tous les trois', 'quelques - uns', 'soi - même', 'messieurs - dames', 'ces messieurs - dames', 'messieurs dames', 'ces messieurs dames', 'les messieurs - dames', 'messieurs - mesdames', 'quelques messieurs - mesdames', 'des messieurs - dames', 'aux messieurs - dames', 'ces messieurs , dames', 'les hommes', 'vous autres', 'tous les hommes', 'vous deux', 'sienne', "l' autre", 'célébrité'
]


    singular_rows = filtered_df[(filtered_df['text'].isin(male_ngram_list))]
    df.loc[singular_rows.index, 'gender'] = 'Male'
    plural_rows = filtered_df[(filtered_df['text'].isin(female_ngram_list))]
    df.loc[plural_rows.index, 'gender'] = 'Female'
    ambiguous_rows = filtered_df[(filtered_df['text'].isin(ambiguous_ngram_list))]
    df.loc[ambiguous_rows.index, 'gender'] = 'Ambiguous'

    del filtered_df
    return df
#%%
def always_gender_head(df):
    filtered_df = df[df['gender'] == 'Not_Assigned']

    always_singular_male_head = ['père', 'mossieur', 'mr.', 'monsignor', 'défunt', 'connétable', 'homme', 'roi', 'mari', 'fils', 'ami', 'enquêteur', 'comte', 'frère', 'duc', 'abbé', 'empereur', 'colonel', 'prince', 'marquis', 'baron', 'vieillard', 'chevalier', 'maître', 'oncle', 'général', 'curé', 'officier', 'petit', 'inconnu', 'compagnon', 'vieux', 'patron', 'cardinal', 'garçon', 'vicomte', 'amant', 'commandant', 'président', 'cocher', 'assassin', 'étranger', 'cousin', 'chef', 'ingénieur', 'ennemi', 'bonhomme', 'prêtre', 'dernier', 'cher', 'malheureux', 'maréchal', 'moine', 'neveu', 'premier', 'lieutenant', 'banquier', 'anglais', 'avocat', 'directeur', 'amiral', 'prisonnier', 'major', 'inspecteur', 'peintre', 'gouverneur', 'brigadier', 'monsieur', 'soldat', 'grand-père', 'blessé', 'magistrat', 'évêque', 'type', 'agent', 'voyageur', 'chasseur', 'époux', 'indien', 'valet', 'messieurs', 'garçons', 'frères', 'fiancé', 'chauffeur', 'paysan', 'fou', 'huissier', 'interlocuteur', 'gardien', 'américain', 'intendant', 'cavalier', 'pape', 'policier', 'portier', 'seigneur', 'gamin', 'fermier', 'gentilhomme', 'gars', 'conducteur', 'marchand', 'canadien', 'jardinier', 'bossu', 'conducteur', 'mort', 'facteur', 'meurtrier', 'geôlier', 'greffier', 'parisien', 'français', 'géant', 'condamné', 'meunier', 'pêcheur', 'nain', 'mendiant', 'tailleur', 'messager', 'juif', 'page', 'caissier', 'pasteur', 'voleur', 'régisseur', 'pharmacien', 'dauphin', 'braconnier', 'forgeron', 'boucher', 'mécanicien', 'sous-préfet', 'fugitif', 'héros', 'clerc', 'caporal', 'chinois', 'berger', 'commandeur', 'sous-officier', 'pâtissier', 'tsar', 'sacristain', 'patient', 'moribond', 'chéri', 'espagnol', 'gendre', 'beau-père', 'chien', 'saint', 'ouvrier', 'passant', 'client', 'dieu', 'innocent', 'employé', 'laquais', 'savant', 'amoureux', 'matelot', 'marin', 'chrétien', 'criminel', 'écrivain', 'espion', 'mourant', 'gaillard', 'serviteur', 'voisin', 'adolescent', 'copain', 'bourgeois', 'allemand', 'écolier', 'visiteur', 'chirurgien', 'insensé', 'vivant', 'mec', 'malfaiteur', 'sorcier', 'coquin', 'italien', 'gentleman', 'protecteur', 'étudiant', 'ambassadeur', 'député', 'acteur', 'imposteur', 'musicien', 'rival', 'guerrier', 'ingrat', 'vagabond', 'scélérat', 'infirmier', 'sot', 'loup', 'souverain', 'second', 'revenant', 'aventurier', 'boulanger', 'comédien', 'européen', 'serrurier', 'sergent', 'beau-frère', 'saint-père', 'beau-frère', 'arrière-grand-père', "grand'père", 'arrière-arrière-grand-père', 'monseigneur', 'granpère', 'pseudo-père', 'accusé', 'préfet', 'aîné', 'homme-là', 'bourreau', 'hôtelier', 'instituteur', 'petit-fils', 'baronnet', 'archevêque', 'lion', 'chat', 'parrain', 'm.', 'papa', 'tuteur', 'sauveur', 'orateur', 'supérieur', 'prieur', 'conseiller', 'sieur', 'chanoine', 'cheval', 'lord', 'sculpteur', 'postillon', 'avoué', 'négociant', 'grand-duc', 'cuisinier', 'bienfaiteur', 'héritier', 'bien-aimé', 'prélat', 'châtelain', 'infortuné', 'tueur', 'sénateur', 'métayer', 'apprenti', 'cadet', 'religieux', 'vainqueur', 'grand-oncle', 'régent', 'venu', 'nègre', 'celui', 'colosse', 'bosseman', 'écuyer', 'chancelier', 'coiffeur', 'irlandais', 'écossais', 'consul', 'orphelin', 'armateur', 'inventeur', 'conteur', 'major', 'un', 'charretier', 'prévôt', 'routier', 'gros', 'bon', 'métis', 'grand', 'seul', 'breton', 'épicier', 'forçat', 'révérend', 'brigand', 'usurier', 'protégé', 'vicaire', 'cabaretier', 'chapelain', 'prévenu', 'barman', 'observateur', 'colaborateur', 'bûcheron', 'satan', 'saint-louis', 'saint-maur', 'saint-simon', 'espagnol', 'pédant', 'soudard', 'rejeton', 'hobereau', 'fluet', 'fripier', 'descendant', 'galant', 'vaillant', 'majordome', 'dormeur', 'trépassé', 'doge', 'tentateur', 'adjoint', 'substitut', 'serin', 'adorateur', 'prétendu', 'promeneur', 'affligé', 'écervelé', 'original', 'paroissien', 'feu', 'factionnaire', 'ancien', 'sémillant', 'singe', 'tigre', 'compère', 'canut', 'parvenu', 'dandy', 'financier', 'malicieux', 'collaborateur', 'farceur', 'absorbé', 'alsacien', 'impertinent', 'valet-de-chambre', 'penseur', 'futur', 'chapelier', 'bottier', 'caudataire', 'musico', 'chanteur', 'banqueroutier', 'danseur', 'bailli', 'praticien', 'sultan', 'correcteur', 'commensal', 'délaissé', 'étourdi', 'capitano', 'marié', 'collégien', 'arrivant', 'créateur', 'dénicheur', 'dévot', 'subordonné', 'jésus', 'laboureur', 'gentil', "grand'papa", 'limonadier', 'obligeant', 'particulier', 'voiturier', 'genevois',  'arménien', 'arrivé', 'bouffon', 'brutal', 'bêta', 'canari', 'canard', 'converti', 'dompté', 'fugitif', 'maladroit', 'minet', 'rédempteur', 'sous-lieutenant', 'turbulent', 'vieux-là'
]
    singular_male_rows = filtered_df[(filtered_df['mention_len'] <= 4)
                            & (filtered_df['number'].isin(['Singular']))
                            & (filtered_df['head_word'].isin(always_singular_male_head))]
    df.loc[singular_male_rows.index, 'gender'] = 'Male'

    always_female_head = ['celle', 'milady','donna', 'duègne', 'mairesse', 'mère', 'soeur', 'femme', 'auteure', 'reine', 'senora', 'fille', 'amie', 'comtesse', 'duchesse', 'princesse', 'maîtresse', 'sœur', 'marquise', 'baronne', 'tante', 'petite', 'inconnue', 'dame', 'veuve', 'cousine', 'fillette', 'nièce', 'servante', 'cousine', 'chère', 'vieille', 'chérie', 'compagne', 'orpheline', 'bonne', 'femmes', 'dames', 'filles', 'malheureuse', 'belle-mère', 'fiancée', 'grand-mère', 'belle-sœur', 'maman', 'petite-fille', 'soeur', "grand'mère", 'belle-fille', 'voisine', 'marraine', 'nourrice', 'gouvernante', 'rivale', 'filleule', 'protégée', 'protectrice', 'patronne', 'bienfaitrice', 'parente', 'cliente', 'belle', 'bien-aimée', 'visiteuse', 'poupée', 'copine', 'prisonnière', 'suivante', 'corsetière', 'coureuse', 'contrôleuse', 'contemplatrice', 'conscrite', 'consoeur', 'compagnonne', 'camériste', 'cadette', 'cuisinière', 'souveraine', 'favorite', 'confidente', 'couturière', 'grand-tante', 'demoiselle', 'danseuse', 'logeuse', 'petite-nièce', 'jumelle', 'conductrice', 'ménagère', 'colaboratrice', 'femelle', 'libératrice', 'captive', 'frangine', 'patiente', 'maitresse', 'gardienne', 'bergère', 'bourgeoise', 'passagère', 'concubine', 'promise', 'demi-sœur', 'muse', 'préférée', 'soubrette', 'supérieure', 'tutrice', 'collaboratrice', 'négresse', 'remplaçante', 'geôlière', 'étrangère', 'morte', 'folle', 'sainte', 'fée', 'religieuse', 'infirmière', 'paysanne', 'actrice', 'française', 'gamine', 'sorcière', 'épouse', 'vierge', 'ouvrière', 'voleuse', 'parisienne', 'institutrice', 'courtisane', 'anglaise', 'seule', 'déesse', 'comédienne', 'coquette', 'chrétienne', 'prostituée', 'mourante', 'criminelle', 'italienne', 'mendiante', 'innocente', 'allemande', 'madone', 'adolescente', 'sirène', 'pute', 'américaine', 'nymphe', 'ennemie', 'héritière', 'amoureuse', 'ingrate', 'pauvresse', 'amante', 'juive', 'martyre', 'coquine', 'bohémienne', 'blonde', 'aventurière', 'vivante', 'serveuse', 'héroïne', 'impératrice', 'brune', 'marchande', 'étudiante', 'aïeule', 'amazone', 'intrigante', 'espagnole', 'indienne', 'chanteuse', 'nonne', 'blanchisseuse', 'pécheresse', 'châtelaine', 'grande', 'gonzesse', 'cantatrice', 'vendeuse', 'insensée', 'chienne', 'vache', 'chinoise', 'gitane', 'lionne', 'gaillarde', 'lady', 'mariée', 'petites', 'amies', 'sœurs', 'demoiselles', 'belles', 'soeurs', 'tantes', 'auditrices', 'lectrices', 'mères', 'innocentes', 'jument', 'chatte', 'nounou', 'correspondante', 'belle-soeur', 'marâtre', 'lectrice', 'mignonne', 'coiffeuse', 'conjointe', "grand'tante", 'poule', 'consolatrice', 'pauline', 'mégère', 'greffière', 'camérière', 'caissière', 'chambrière', 'cavalière', 'première', 'mercière', 'meurtrière', 'lavandière', 'régulière', 'banquière', 'barbière', 'nourricière', 'particulière', 'sous-caissière', 'chevalière', 'chevrière', 'coéquipière', 'crémière', 'fermière', 'persécutrice', 'coadjutrice', 'directrice', 'génitrice', 'fondatrice', 'débitrice', 'dominatrice', 'dénonciatrice', 'détentrice', 'calomniatrice', 'créatrice', 'modératrice', 'rédactrice', 'spectatrice', 'présentatrice', 'interlocutrice', 'inspectrice', 'ambassadrice', 'traductrice', 'narratrice', 'ex-institutrice', 'réalisatrice', 'donatrice', 'opératrice', 'observatrice', 'alice', 'testatrice', 'tentatrice', 'éducatrice', 'éditrice', 'animatrice', 'organisatrice', 'triomphatrice', 'accusatrice', 'accompagnatrice', 'usurpatrice', 'admiratrice', 'factrice', 'enquêtrice', 'perceptrice', 'séductrice', 'navigatrice', 'simulatrice', 'monitrice', 'manipulatrice', 'auditrice', 'prestidigitatrice', 'sculptrice', 'répétitrice', 'prédatrice', 'préparatrice', 'conjuratrice', 'affabulatrice', 'incitatrice', 'inspiratrice', 'indicatrice', 'novatrice', 'délatrice', 'eurydice', 'employée', 'invitée', 'infortunée', 'possédée', 'habituée', 'pestiférée', 'exilée', 'noyée', 'condamnée', 'alliée', 'désespérée', 'naufragée', 'ressuscitée', 'traînée', 'accusée', 'damnée', 'divorcée', 'enragée', 'abandonnée', 'salariée', 'mijaurée', 'réprouvée', 'dévergondée', 'croisée', 'effrontée', 'droguée', 'déclassée', 'blessée', 'privilégiée', 'accouchée', 'épousée', 'passionnée', 'isolée', 'névrosée', 'brûlée', 'toquée', 'suicidée', 'réfugiée', 'émigrée', 'évaporée', 'inculpée', 'illuminée', 'handicapée', 'associée', 'agitée', 'affamée', 'aînée', 'entêtée', 'exaltée', 'défroquée', 'suppliciée', 'désœuvrée', 'écervelée', 'cinglée', 'adorée', 'athée', 'détraquée', 'dulcinée', 'délurée', 'demeurée', 'opprimée', 'passante', 'voyante', 'méchante', 'débutante', 'savante', 'surveillante', 'géante', 'convalescente', 'figurante', 'commerçante', 'vaillante', 'assistante', 'absente', 'infante', 'protestante', 'revenante', 'pénitente', 'insolente', 'indifférente', 'ignorante', 'élégante', 'suppliante', 'agonisante', 'croyante', 'charmante', 'postulante', 'présidente', 'démente', 'impertinente', 'indépendante', 'sympathisante', 'feignante', 'estivante', 'extravagante', 'gisante', 'gérante', 'arrivante', 'adjointe', 'enseignante', 'militante', 'imprudente', 'bacchante', 'soignante', 'fainéante', 'impudente', 'descendante', 'combattante', 'communiante', 'hôtesse', 'tigresse', 'enchanteresse', 'abbesse', 'chanoinesse', 'petite-maîtresse', 'archiduchesse', 'prêtresse', 'sous-maîtresse', 'drôlesse', 'diablesse', 'mulâtresse', 'doctoresse', 'ogresse', 'vicomtesse', 'devineresse', 'prophétesse', 'ivrognesse', 'poétesse', 'traîtresse', 'sauvagesse', 'chasseresse', 'druidesse', 'grande-duchesse', 'quakeresse', 'scélératesse', 'papesse', 'turquesse', 'stewardesse', 'vieillesse', 'ânesse', 'jésuitesse', 'peintresse', 'demi-déesse', 'contremaîtresse', 'docteresse', 'dernière', 'madame', 'gueuse', 'garce', 'salope', 'conne', 'souris', 'petite-là', 'marie', 'madeleine', 'vipère', 'idiote', 'catin', 'pimbêche', 'intruse', 'meuf', 'donzelle', 'orgueilleuse', 'vieille-là', 'françaises', 'saintes', 'compagnes', 'maréchale', 'louve', 'mademoiselle', 'douairière', 'meunière', 'mlle', 'mme', 'magicienne', 'une', 'venue', 'agée', 'grosse', 'métisse', 'seconde', 'protégée', 'défunte', 'voyageuse', 'dompteuse', 'ouvreuse', 'dormeuse', 'revendeuse', 'causeuse', 'empoisonneuse', 'procureuse', 'receveuse', 'goualeuse', 'rapporteuse', 'promeneuse', 'plaideuse', 'charmeuse', 'quêteuse', 'repasseuse', 'habilleuse', 'fileuse', 'brûleuse', 'brodeuse', 'placeuse', 'laveuse', 'baigneuse', 'maquilleuse', 'menteuse', 'rabouilleuse', 'boiteuse', 'curieuse', 'masseuse', 'chroniqueuse', 'magnétiseuse', 'tueuse', 'travailleuse', 'nageuse', 'paresseuse', 'régisseuse', 'balayeuse', 'solliciteuse', 'chercheuse', 'brocanteuse', 'bienheureuse', 'nourrisseuse', 'tricoteuse', 'teuse', 'conteuse', 'réveilleuse', 'pleureuse', 'boiteuse', 'parfumeuse', 'loueuse', 'tapeuse', 'rêveuse', 'fumeuse', 'pêcheuse', 'fosseuse', 'tailleuse', 'sainte-marie', 'espagnole', 'miss', 'pédante', 'donna', 'gitana', 'rejetonne', 'hoberelle', 'fluette', 'fripière', 'descendante', 'galante', 'vaillante', 'goule', 'intendante', 'gouvernante', 'dormeuse', 'trépassée', 'tentatrice', 'provinciale', 'recluse', 'lingère', 'adjointe', 'substitute', 'serine', 'adoratrice', 'prétendue', 'promeneuse', 'affligée', 'écervelée', 'originale', 'matrone', 'paroissienne', 'feue', 'ancienne', 'sémillante', 'canuse', 'parvenue', 'malicieuse', 'collaboratrice', 'absorbée', 'alsacienne', 'impertinente', 'future', 'chapelière', 'bottière', 'chanteuse', 'banqueroutière', 'danseuse', 'harpie', 'baillie', 'praticienne', 'sultane', 'correctrice', 'commensale', 'délaissée', 'étourdie', 'mariée', 'collégienne', 'arrivante', 'créatrice', 'dénicheuse', 'dévote', 'dauphine', 'subordonnée', 'maraîchère', 'laboureuse', 'gentille', "grand'maman", 'limonadière', 'obligeante', 'particulière', 'voiturière', 'genevoise', 'arménienne', 'arrivée', 'brutale', 'bêtasse', 'convertie', 'domptée', 'fugitive', 'maladroite', 'minette', 'née', 'rédemptrice', 'turbulente', 'écolière'

]
    female_rows = filtered_df[(filtered_df['mention_len'] <= 4)
                            & (filtered_df['number'].isin(['Singular']))
                            & (filtered_df['head_word'].isin(always_female_head))]
    df.loc[female_rows.index, 'gender'] = 'Female'


    always_ambiguous_head = ['gendarmerie', 'cortége', 'hôte', 'sainteté', 'seigneurie', 'chœur', 'altesse', 'ange', 'armée', 'assemblée', 'assistance', 'auditoire', 'bande', 'bataillon', 'bateau', 'belle-famille', 'bourg', 'brigade', 'brute', 'bébé', 'bête', 'cadavre', 'dépouille', 'camp', 'canaille', 'capitale', 'caravane', 'cavalerie', 'centaine', 'cinquantaine', 'cité', 'clergé', 'clientèle', 'cohorte', 'colonie', 'communauté', 'commune', 'commère', 'compagnie', 'comté', 'confrérie', 'connaissance', 'conseil', 'contrée', 'corporation', 'corps', 'cortège', 'couple', 'cour', 'couvent', 'couvée', 'crapule', 'créature', 'demi-douzaine', 'diocèse', 'divinité', 'division', 'dizaine', 'douzaine', 'démon', 'département', 'entourage', 'escadron', 'escorte', 'famille', 'fantôme', 'foule', 'garnison', 'garçonnière', 'gens', 'gentilhommière', 'gouvernement', 'groupe', 'horde', 'idole', 'individu', 'jeunesse', 'justice', 'milice', 'ministère', 'monastère', 'monde', 'monstre', 'multitude', 'nation', 'noblesse', 'nombreuses', 'nombreux', 'ombre', 'orchestre', 'ordre', 'parent', 'paroisse', 'patrouille', 'pays', 'personnage', 'personne', 'personnel', 'peuple', 'plupart', 'police', 'populace', 'population', 'province', 'public', 'quartier', 'quatrième', 'quinzaine', 'race', 'royaume', 'régiment', 'république', 'secteur', 'section', 'sentinelle', 'silhouette', 'société', 'spectre', 'star', 'statue', 'tablée', 'tribu', 'troupe', 'vermine', 'village', 'ville', 'vingtaine', 'âme', 'équipage', 'être', 'autres', 'personnes', 'habitants', 'compatriotes', 'enfants', 'compagnons', 'amis', 'soldats', 'parents', 'voyageurs', 'gens', 'monde', 'morts', 'domestiques', 'paysans', 'camarades', 'gendarmes', 'passants', 'passants', 'ouvriers', 'médecins', 'prisonniers', 'indiens', 'hôtes', 'allemands', 'officiers', 'pauvres', 'prussiens', 'ennemis', 'matelots', 'chiens', 'siens', 'convives', 'étrangers', 'flics', 'juges', 'gardes', 'assistants', 'invités', 'voisins', 'chasseurs', 'russes', 'clients', 'collègues', 'agents', 'maîtres', 'malades', 'bandits', 'vivants', 'voleurs', 'blessés', 'animal', 'ménage', 'majesté', 'victime', 'moi', 'lecteur', 'oiseau', 'amour', 'ours', 'personnages', 'colons', 'jumeaux'
]
    always_ambiguous_rows = filtered_df[(filtered_df['mention_len'] <= 4)
                                        & (filtered_df['head_word'].isin(always_ambiguous_head))]
    df.loc[always_ambiguous_rows.index, 'gender'] = 'Ambiguous'

    del filtered_df
    return df
#%%
def singular_ambiguous_gender_head(df):
    filtered_df = df.copy()

    # heads that can either be male or female. The context is used to disambiguate
    singular_ambiguous_head = ['enfant', 'reporter', 'autre', 'docteur', 'détective', 'capitaine', 'complice', 'médecin', 'commissaire', 'notaire', 'juge', 'artiste', 'domestique', 'malade', 'maire', 'concierge', 'ministre', 'professeur', 'auteur', 'esclave', 'troisième', 'coupable', 'misérable', 'pensionnaire', 'élève', 'somnambule', 'garde-malade', 'garde', 'secrétaire', 'modiste', 'camarade', 'chèvre', 'gosse', 'aveugle', 'aristocrate', 'gendarme', 'poète', 'commissionnaire', 'traître', 'lâche', 'guide', 'flic', 'témoin', 'philosophe', 'imbécile', 'sauvage', 'bandit', 'ivrogne', 'journaliste', 'militaire', 'pauvre', 'novice', 'russe', 'collègue', 'fleuriste', 'aubergiste', 'adversaire', 'propriétaire', 'môme', 'architecte', 'pilote', 'ermite', 'pote', 'interne', 'suisse', 'procureur', 'infirme', 'jeune', 'prophète', 'brave', 'drôle', 'monarque', 'mousquetaire', 'vétérinaire', 'poëte', 'photographe', 'apothicaire', 'matamore', 'apôtre', 'séminariste', 'commis', 'noble', 'drapier', 'fat', 'diable', 'diplomate', 'droguiste', 'paralytique', 'journalier', 'aide-de-camp', 'partenaire', 'dentiste', 'confrère', 'copiste', 'centenaire', 'timide', 'vaguemestre', 'patriote', 'convive', 'ex-jésuite', 'gnangnan', 'anatomiste', 'dr', 'fidèle', 'myope', 'proviseur'
                               ]

    vowel_singular_ambiguous_head = [head for head in singular_ambiguous_head if head.startswith(('a', 'e', 'é', 'è', 'ê', 'i', 'o', 'u', 'y', 'h'))]
    consonant_singular_ambiguous_head = [head for head in singular_ambiguous_head if head not in vowel_singular_ambiguous_head]

    male_vowel_prefix = ['au ', 'du ',  'un ', 'cet ', 'ce ', 'le ', 'cher', 'petit', 'premier', 'second', 'dernier', 'bel', 'ancien', 'malheureux', 'grand', 'intéressant', 'nouvel', 'quel', 'audacieux', 'heureux', 'infortuné ', 'le très']
    male_vowel_prefix += [prefix + ' ' + adj + ' ' for prefix in ['mon', 'ton', 'son', "l'", 'quel'] for adj in ['cher', 'petit', 'premier', 'second', 'troisième', 'dernier', 'pauvre', 'propre', 'brave', 'bel', 'ancien', 'jeune', 'malheureux', 'redoutable', 'grand', 'intéressant', 'nouvel', 'terrible', 'seul', 'excellent', 'fidèle']]
    male_vowel_prefix += [prefix + ' ' + adj + ' ' for prefix in ['notre', 'votre', 'leur', 'pauvre'] for adj in ['cher', 'petit', 'premier', 'second', 'dernier', 'bel', 'ancien', 'malheureux', 'grand', 'intéressant', 'nouvel', 'seul', 'excellent']]

    female_vowel_prefix = ['la ', 'cette', 'une', 'ma ', 'sa ', 'chère', 'petite', 'première', 'seconde', 'dernière', 'belle', 'ancienne', 'malheureuse', 'grande', 'innocente', 'intéressante', 'nouvelle', 'seule', 'excellente']
    female_vowel_prefix += [prefix + ' ' + adj + ' ' for prefix in ['notre', 'votre', 'leur', 'pauvre', "l'", 'quelle'] for adj in ['chère', 'petite', 'première', 'seconde', 'dernière', 'belle', 'ancienne', 'malheureuse', 'grande', 'innocente', 'intéressante', 'nouvelle', 'seule', 'excellente']]

    male_rows = filtered_df[(filtered_df['mention_len'] <= 4)
                            & (filtered_df['number'].isin(['Singular']))
                            & (filtered_df['gender'].isin(['Not_Assigned']))
                            & (filtered_df['text'].str.startswith(tuple(male_vowel_prefix)))
                            & (filtered_df['head_word'].isin(vowel_singular_ambiguous_head))]
    filtered_df.loc[male_rows.index, 'gender'] = 'Male'

    female_rows = filtered_df[(filtered_df['mention_len'] <= 4)
                            & (filtered_df['gender'].isin(['Not_Assigned']))
                            & (filtered_df['number'].isin(['Singular']))
                            & (filtered_df['text'].str.startswith(tuple(female_vowel_prefix)))
                            & (filtered_df['head_word'].isin(vowel_singular_ambiguous_head))]
    filtered_df.loc[female_rows.index, 'gender'] = 'Female'

    # -------consonant--ambiguous--head--------------------------------

    male_consonant_prefix = ['le ', 'du ', 'au ', 'un ', 'ce ', 'cet ', 'mon ', 'messire', 'ton ', 'son ', 'cher ', "l' ancien ", 'm. ', 'mm .', 'mm ', 'monsieur',  'quel ', 'cher ', 'petit ', 'premier ', 'second ', 'dernier ', 'bel ', 'ancien ', 'malheureux', 'grand ', 'intéressant ', 'nouveau', 'seul ', 'excellent ', 'bon ', 'audacieux ', 'heureux ', 'infortuné ', 'vieux ']
    male_consonant_prefix += [prefix + ' ' + adj + ' ' for prefix in ['notre', 'votre', 'leur', 'pauvre', "l'"] for adj in ['cher', 'petit', 'premier', 'second', 'dernier', 'bel', 'ancien', 'malheureux', 'grand', 'intéressant', 'nouvel', 'seul', 'excellent', 'bon', 'audacieux', 'heureux', 'infortuné', 'vieux']]
    male_rows = filtered_df[(filtered_df['mention_len'] <= 4)
                            &(filtered_df['mention_len'] > 1)
                            & (filtered_df['number'].isin(['Singular']))
                            & (filtered_df['gender'].isin(['Not_Assigned']))
                            & (filtered_df['text'].str.startswith(tuple(male_consonant_prefix)))
                            & (filtered_df['head_word'].isin(consonant_singular_ambiguous_head))]
    filtered_df.loc[male_rows.index, 'gender'] = 'Male'

    female_consonant_prefix = ['la ', 'cette', 'une ', 'ma ', 'ta ', 'sa ', 'chère ', "l' ancienne ", 'quelle ', 'chère', 'petite', 'première', 'seconde', 'dernière', 'belle', 'ancienne', 'malheureuse', 'grande', 'innocente', 'intéressante', 'nouvelle', 'seule', 'excellente', 'bonne']
    female_consonant_prefix += [prefix + ' ' + adj + ' ' for prefix in ['notre', 'votre', 'leur', 'pauvre', "l'", 'quelle'] for adj in ['chère', 'petite', 'première', 'seconde', 'dernière', 'belle', 'ancienne', 'malheureuse', 'grande', 'innocente', 'intéressante', 'nouvelle', 'seule', 'excellente', 'bonne']]
    female_rows = filtered_df[(filtered_df['mention_len'] <= 4)
                            &(filtered_df['mention_len'] > 1)
                            & (filtered_df['number'].isin(['Singular']))
                            & (filtered_df['gender'].isin(['Not_Assigned']))
                            & (filtered_df['text'].str.startswith(tuple(female_consonant_prefix)))
                            & (filtered_df['head_word'].isin(consonant_singular_ambiguous_head))]
    filtered_df.loc[female_rows.index, 'gender'] = 'Female'

    ambiguous_rows = filtered_df[(filtered_df['mention_len'] <= 4)
                            & (filtered_df['gender'].isin(['Not_Assigned']))
                            & (filtered_df['number'].isin(['Singular']))
                            & (filtered_df['head_word'].isin(singular_ambiguous_head))]
    filtered_df.loc[ambiguous_rows.index, 'gender'] = 'Ambiguous'

    del df
    return filtered_df
#%%
def assign_gender_to_plural_head(df):
    filtered_df = df
    filtered_df = filtered_df[filtered_df['number'] == 'Plural']
    filtered_df = filtered_df[~(filtered_df['text'].str.contains(' et '))]
    filtered_df = filtered_df[(filtered_df['mention_len'] <= 4)]

    always_plural_female_head = ['femmes', 'dames', 'filles', 'sœurs', 'amies', 'mères', 'demoiselles', 'compagnes', 'elles', 'religieuses', 'servantes', 'unes', 'maîtresses', 'princesses', 'fillettes', 'religieuses', 'petites', 'cousines', 'ouvrières', 'vieilles', 'voisines', 'danseuses', 'malheureuses', 'voyageuses', 'visiteuses', 'vendeuses', 'serveuses', 'promeneuses', 'ouvreuses', 'laveuses', 'blanchisseuses', 'chanteuses', 'voleuses', 'baigneuses', 'amoureuses', 'faneuses', 'curieuses', 'entraîneuses', 'brodeuses', 'travailleuses', 'pleureuses', 'gueuses', 'tricoteuses', 'ténébreuses', 'dormeuses', 'joueuses', 'acheteuses', 'revendeuses', 'vendangeuses', 'européennes', 'lycéennes', 'méditerranéennes', 'chicoréennes', 'vendéennes', 'dahoméennes', 'fées', 'prostituées', 'poupées', 'infortunées', 'aînées', 'protégées', 'invitées', 'condamnées', 'fiancées', 'mariées', 'employées', 'pygmées', 'épousées', 'habituées', 'âgées', 'accouchées', 'athées', 'privilégiées', 'machabées', 'subordonnées', 'intéressées', 'aimées', 'tantes', 'suivantes', 'arrivantes', 'gouvernantes', 'surveillantes', 'passantes', 'élégantes', 'habitantes', 'vivantes', 'amantes', 'aspirantes', 'étudiantes', 'jumelles']

    plural_female_rows = filtered_df[(filtered_df['head_word'].isin(always_plural_female_head))]
    df.loc[plural_female_rows.index, 'gender'] = 'Female'

    always_plural_male_head = ['garçons', 'frères', 'gars', 'fils', 'prêtres', 'moines', 'pères', 'mecs', 'gentilshommes', 'maris', 'messieurs', 'oncles']
    plural_male_rows = filtered_df[(filtered_df['head_word'].isin(always_plural_male_head))]
    df.loc[plural_male_rows.index, 'gender'] = 'Male'


    always_plural_ambiguous_head = ['anglais', 'masse', 'français', 'élèves', 'cavaliers', 'témoins', 'malheureux', 'spectateurs', 'visiteurs', 'bourgeois', 'derniers', 'espagnols', 'confrères', 'êtres', 'sentinelles', 'époux', 'adversaires', 'blancs', 'étudiants', 'jurés', 'personnages','ceux', 'deux', 'rois', 'petits', 'rois', 'vieux', 'premiers', 'riches', 'fugitifs', 'complices', 'curieux', 'quatre', 'anciens', 'jeunes', 'artistes', 'arabes', 'lecteurs', 'marins', 'princes', 'juifs', 'passagers', 'vieillards', 'dieux', 'ancêtres', 'eux', 'anges', 'magistrats', 'parisiens', 'misérables', 'chefs', 'américains', 'coupables', 'amoureux', 'journalistes', 'sauvages', 'chrétiens', 'policiers', 'colons', 'ministres', 'romains', 'employés', 'pêcheurs', 'musiciens', 'assassins', 'loups', 'amants', 'nôtre', 'brigands', 'turcs', 'valets', 'fidèles', 'chinois', 'marchands', 'serviteurs', 'sujets', 'citoyens', 'copains', 'promeneurs', 'troupes', 'poètes', 'assaillants', 'semblables', 'gardiens', 'grecs', 'joueurs', 'beaucoup', 'victimes', 'acteurs', 'touristes', 'républicains', 'cousins', 'familles', 'groupes', 'indigènes', 'savants', 'animaux', 'condamnés', 'grands', 'auditeurs', 'gosses', 'venus', 'gamins', 'esclaves', 'noirs', 'combattants', 'esclaves', 'militaires', 'nobles', 'guerriers', 'inspecteurs', 'saints', 'peintres', 'autrichiens', 'porteurs', 'petites', 'auteurs', 'courtisans', 'pompiers', 'saints', 'barbares', 'européens', 'nègres', 'adultes', 'seigneurs', 'insurgés', 'archers', 'inconnus', 'royalistes', 'types', 'peuples', 'pirates', 'oiseaux', 'bretons', 'rebelles', 'poissons', 'coquins', 'italiens', 'neveux', 'jésuites', 'assiégés', 'généraux', 'danseurs', 'aïeux', 'fous', 'conjurés', 'trois', 'travailleurs', 'locataires', 'bourreaux', 'fous', 'dragons', 'suisses', 'députés', 'professeurs', 'desquels', '-là', 'patriotes', 'avocats', 'chevaliers', 'vainqueurs', 'couples', 'propriétaires', 'contemporains', 'arrivants', 'écoliers', 'concitoyens', 'fiancés', 'partisans', 'fuyards', 'forçats', 'armées', 'couvées', 'bourgeoisie']
    plural_ambiguous_rows = filtered_df[(filtered_df['head_word'].isin(always_plural_ambiguous_head))]
    df.loc[plural_ambiguous_rows.index, 'gender'] = 'Ambiguous'

    return df
#%%
def assign_gender_to_singular_proper_mentions(entities_df):
    filtered_df = entities_df[entities_df['gender'] == 'Not_Assigned']
    filtered_df = filtered_df[filtered_df['prop'] == 'PROP']
    filtered_df = filtered_df[(filtered_df['number'].isin(['Singular']))]

    male_head_list = ['m.', 'monsieur', 'mossieur', 'monseigneur', 'don', 'sir', 'lord', 'maître', 'saint', 'messire', 'père', 'sieur', 'don', 'dom', 'comte', 'mr', 'ami', 'nommé', 'dénommé', 'dieu', 'marseillais', 'canadien', 'page', 'mgr', 'mister', 'mr.', 'baron', 'seigneur', 'frère', 'jésus', 'marquis', 'prince', 'celui', 'pasteur', 'papa', 'roi', 'master', 'herr', 'oncle', 'milord', 'mylord', 'bien-aimé', 'duc', 'vicomte', 'sire', 'abbé', 'cousin', 'parisien', 'président', 'señor', 'fils', 'tonton', 'maréchal', 'juif', 'bonhomme', 'sergent', 'captain', 'cheikh', 'christ']
    male_proper_mentions_rows = filtered_df[(filtered_df['head_word'].isin(male_head_list))]
    entities_df.loc[male_proper_mentions_rows.index, 'gender'] = 'Male'

    female_head_list = ['miss', 'mme', 'mlle', 'madame', 'mademoiselle', 'veuve', 'vierge', 'lady', 'dame', 'tante', 'mistress', 'mrs', 'mère', 'maman', 'nommée', 'dénommée', 'sœur', 'chère', 'signora', 'notre-dame', "mam'", 'sainte', 'marquise', 'reine', 'demoiselle', 'comtesse', 'bien-aimée', 'princesse', 'belle', 'petite', 'cousine', 'fille', 'parisienne', 'frau', 'duchesse', 'juive', 'elle-même', 'doña', 'dona']
    female_proper_mentions_rows = filtered_df[(filtered_df['head_word'].isin(female_head_list))]
    entities_df.loc[female_proper_mentions_rows.index, 'gender'] = 'Female'

    male_prefix_list = ['m.', 'monsieur', 'mossieur', 'monseigneur', 'don', 'sir', 'lord', 'maître', 'saint', 'messire', 'père', 'sieur', 'dom', 'comte', 'mr', 'ami', 'nommé', 'dénommé', 'dieu', 'marseillais', 'canadien', 'page', 'mgr', 'mister', 'mr.', 'baron', 'seigneur', 'frère', 'jésus', 'marquis', 'prince', 'celui', 'pasteur', 'papa', 'roi', 'master', 'herr', 'oncle', 'milord', 'mylord', 'bien-aimé', 'duc', 'cheikh', 'vicomte', 'sire', 'abbé', 'cousin', 'parisien', 'président', 'señor', 'fils', 'tonton', 'maréchal', 'juif', 'bonhomme', 'le petit', 'le vieux', 'mon petit', 'mon cher', 'le beau', 'le chanoine', 'le gros', 'mon pauvre', 'le pauvre', 'le grand', 'le captain', 'le dr', "l' évêque", 'le bel' , 'le bon', 'le cheikh', 'le jeune', 'le vieil', 'du petit', 'notre cher', 'votre cher', 'ce digne']
    male_prefix_list = [prefix + ' ' for prefix in male_prefix_list]
    male_proper_mentions_rows = filtered_df[(filtered_df['text'].str.startswith(tuple((male_prefix_list))))]
    entities_df.loc[male_proper_mentions_rows.index, 'gender'] = 'Male'


    female_prefix_list = ['miss', 'mme', 'mlle', 'madame', 'mademoiselle', 'veuve', 'vierge', 'lady', 'dame', 'tante', 'mistress', 'mrs', 'mère', 'maman', 'nommée', 'dénommée', 'sœur', 'chère', 'signora', 'notre-dame', "mam'", 'sainte', 'marquise', 'reine', 'demoiselle', 'comtesse', 'bien-aimée', 'princesse', 'belle', 'petite', 'cousine', 'fille', 'parisienne', 'frau', 'duchesse', 'juive', 'elle-même', 'la petite', 'la vieille', 'ma petite', 'ma chère', 'la belle', 'la mémé', 'ma pauvre', 'la pauvre', 'la grande', 'douairière', 'la belle', 'la jeune', 'la bonne', 'votre chère', 'notre chère', 'la bonne']
    female_prefix_list = [prefix + ' ' for prefix in female_prefix_list]
    female_proper_mentions_rows = filtered_df[(filtered_df['text'].str.startswith(tuple((female_prefix_list))))]
    entities_df.loc[female_proper_mentions_rows.index, 'gender'] = 'Female'

    del filtered_df
    return entities_df
#%%
def assign_gender_to_complex_proper_mentions(df):

    filtered_df = df[(df['gender'] == 'Not_Assigned')]
    filtered_df = filtered_df[(filtered_df['prop'] == 'PROP')]
    filtered_df = filtered_df[(filtered_df['number'] == 'Singular')]
    # filtered_df = filtered_df[(filtered_df['mention_len'] >= 3)]

    nobiliary_particle = ['de', "d'", 'des', 'du', 'af', 'von', 'le', 'la']

    female_singular_adj = ['adroite', 'affreuse', 'ancienne', 'audacieuse', 'avisée', 'belle', 'bien nommée', 'blonde', 'bonne', 'brillante', 'brutale', 'bruyante', 'charmante', 'chère', 'courageuse', 'curieuse', 'damnée', 'dangereuse', 'dernière', 'diablotine', 'divine', 'doucereuse', 'douce', 'délicieuse', 'dénommée', 'effrayante', 'ennuyeuse', 'enragée', 'excellente', 'fameuse', 'fausse', 'fichue', 'fière', 'fringuante', 'froide', 'galante', 'gentille', 'grande', 'grosse', 'généreuse', 'hardie', 'heureuse', 'hideuse', 'idiote', 'immortelle', 'impatiente', 'imprudente', 'infecte', 'infernale', 'infortunée', 'ingrate', 'innocente', 'inquiétante', 'insolente', 'insoucieuse', 'insurgée', 'intéressante', 'joyeuse', 'mme.', 'majestueuse', 'maladroite', 'malencontreuse', 'malheureuse', 'maudite', 'mauvaise', 'merveilleuse', 'monstrueuse', 'mlle.', 'mystérieuse', 'méchante', 'nommée', 'nonchalante', 'nouvelle', 'obligeante', 'obscure', 'odieuse', 'orgueilleuse', 'originale', 'petite', 'pieuse', 'piteuse', 'plaintive', 'polissonne', 'première', 'prestigieuse', 'prodigieuse', 'précieuse', 'prétendue', 'prétentieuse', 'puissante', 'ravissante', 'repoussante', 'resplendissante', 'répugnante', 'sacrée', 'salope', 'satanée', 'savante', 'seconde', 'singulière', 'sournoise', 'subtile', 'séduisante', 'valeureuse', 'vertueuse', 'vieille', 'vilaine', 'éloquente', 'élégante', 'éternelle', 'étincelante', 'étonnante', 'étourdissante', 'conquérante', 'folle', 'certaine', 'toute', 'bienheureuse', 'glorieuse', 'jolie', 'citoyenne', 'évanescente', 'brune', 'morte', 'mère', 'prudente', 'amie', 'entière']

    male_singular_adj = ['évanescent', 'adroit', 'affreux', 'ancien', 'audacieux', 'avisé', 'beau', 'bel', 'bien nommé', 'blond', 'bon', 'brillant', 'bienheureux', 'brutal', 'certain', 'bruyant', 'charmant', 'cher', 'courageux', 'curieux', 'damné', 'dangereux', 'dernier', 'diablotin', 'divin', 'doucereux', 'doux', 'délicieux', 'dénommé', 'effrayant', 'ennuyeux', 'enragé', 'excellent', 'fameux',  'faux', 'fichu', 'fier', 'fringuant', 'froid', 'galant', 'gentil', 'grand', 'gros', 'généreux', 'hardi', 'heureux', 'hideux', 'idiot', 'immortel', 'impatient', 'imprudent', 'infect', 'infernal', 'infortuné', 'ingrat', 'innocent', 'inquiétant', 'insolent', 'insoucieux', 'insurgé', 'intéressant', 'joyeux', 'm.', 'majestueux', 'maladroit', 'malencontreux', 'malheureux', 'maudit', 'mauvais', 'merveilleux', 'monstrueux', 'mr.', 'mystérieux', 'méchant', 'nommé', 'nonchalant', 'nouveau', 'nouvel', 'obligeant', 'obscur', 'odieux', 'orgueilleux', 'original', 'petit', 'pieux', 'piteux', 'plaintif', 'polisson', 'premier', 'prestigieux', 'prodigieux', 'précieux', 'prétendu', 'prétentieux', 'puissant', 'ravissant', 'repoussant', 'resplendissant', 'répugnant', 'sacré', 'salaud', 'salop', 'satané', 'savant', 'second', 'singulier', 'sournois', 'subtil', 'séduisant', 'valeureux', 'vertueux', 'vieil', 'vieux', 'vilain', 'éloquent', 'élégant', 'éternel', 'étincelant', 'étonnant', 'étourdissant', 'conquérant', 'fou', 'glorieux', 'citoyen', 'mort', 'père', 'prudent', 'ami', 'entier', 'singe', 'diable']

    ambiguous_singular_adj = ['farouche', 'abominable', 'absurde', 'admirable', 'adorable', 'aimable', 'austère', 'autre', 'avare', 'bizarre', 'brave', 'célèbre', 'docile', 'docte', 'déplorable', 'détestable', 'estimable', 'extraordinaire', 'exécrable', 'fade', 'flegmatique', 'frénétique', 'honnête', 'horrible', 'humble', 'hypocrite', 'ignoble', 'illustre', 'immense', 'immonde', 'impassible', 'improbable', 'incomparable', 'inepte', 'infâme', 'inqualifiable', 'insatiable', 'insupportable', 'intrépide', 'invincible', 'invisible', 'invraisemblable', 'irascible', 'irréprochable', 'jeune', 'juste', 'malhonnête', 'maussade', 'misérable', 'méprisable', 'même', 'noble', 'pauvre', 'pitoyable', 'poétique', 'pâle', 'redoutable', 'respectable', 'ridicule', 'robuste', 'rude', 'rustique', 'sage', 'sale', 'sauvage', 'sceptique', 'sensible', 'sinistre', 'sombre', 'stoïque', 'stupide', 'sublime', 'superbe', 'sympathique', 'terrible', 'timide', 'triste', 'vulgaire', 'énigmatique', 'énorme', 'épouvantable', 'étrange', 'fantasque', 'fantastique', 'ci-devant', 'sévère', 'simple', 'ci - devant', 'trop', 'moderne',]


    male_prefix = ['le', 'ce', 'cet', 'du', 'un', 'au']
    female_prefix = ['la', 'cette', 'ma', 'ta', 'sa', 'une', 'à la']
    ambiguous_prefix = ["l'", 'notre', 'votre', 'leur', 'mon', 'ton', 'son', 'pauvre', 'tout']


    female_attributes = [f'{prefix} {adj}' for prefix in female_prefix for adj in female_singular_adj + ambiguous_singular_adj ] + [f'{prefix} {adj}' for prefix in ambiguous_prefix for adj in female_singular_adj]
    female_rows = filtered_df[(filtered_df['gender'] == 'Not_Assigned')&
                            ((filtered_df['text'].str.contains('|'.join(map(re.escape, female_attributes)), regex=True))|(filtered_df['text'].str.startswith(tuple(['cette ', 'ma ', 'ta ', 'sa ', 'chère ', 'une ']))))]
    df.loc[female_rows.index, 'gender'] = 'Female'
    filtered_df.loc[female_rows.index, 'gender'] = 'Female'


    male_attributes = [f'{prefix} {adj}' for prefix in male_prefix for adj in male_singular_adj + ambiguous_singular_adj ] + [f'{prefix} {adj}' for prefix in ambiguous_prefix for adj in male_singular_adj]
    male_rows = filtered_df[(filtered_df['gender'] == 'Not_Assigned')&
                            ((filtered_df['text'].str.contains('|'.join(map(re.escape, male_attributes)), regex=True))|(filtered_df['text'].str.startswith(tuple(['un ', 'ce ', 'cet ', 'cher ']))))]
    df.loc[male_rows.index, 'gender'] = 'Male'

    # -- Assign gender based on suffix

    male_suffix = ['seul', "lui-même"]
    male_suffix = [f" {suffix}" for suffix in male_suffix]
    male_rows = filtered_df[(filtered_df['gender'] == 'Not_Assigned')
                            & (filtered_df['mention_len'] <= 4)
                            & (filtered_df['text'].str.endswith(tuple(male_suffix)))]
    df.loc[male_rows.index, 'gender'] = 'Male'

    return df
#%%
def assign_gender_to_proper_mentions_from_gendered_mentions(df):

    all_gendered_per_entities = df[(df['cat'] == 'PER')
                               &(df["number"] == "Singular")
                               &(df['gender'].isin(["Male", "Female"]))][['text', 'number', 'gender', 'mention_len']]

    to_gender_proper_singular = df[(df['cat'] == 'PER')
                                   &(df["number"] == "Singular")
                                   &(df["prop"] == "PROP")
                                   & ~(df['text'].str.contains(f' et '))
                                   &(df['gender'].isin(["Not_Assigned"]))][['text', 'head_word', 'mention_len', 'gender']]

    male_count_list = []
    female_count_list = []

    to_gender_proper_singular.drop_duplicates(inplace=True)

    for proper_text, proper_head in to_gender_proper_singular[['text', 'head_word']].values:
        if proper_head == "-":
            proper_head = proper_text
        if proper_text == "d' "+ proper_head:
            proper_head = proper_text

        # print(proper_head)
        proper_head = proper_head.replace('\\', '')
        # Create the regex pattern to match the whole word
        proper_head_boundaries = rf'(?<!-)\b{re.escape(proper_head)}\b(?!-)'
        # print(proper_head)
        proper_filtered = all_gendered_per_entities[(all_gendered_per_entities['number'] == 'Singular')
                             &(all_gendered_per_entities['text'].str.contains(proper_head_boundaries, regex=True))
                             & ~(all_gendered_per_entities['text'].str.contains(rf' et ')) &
                                                  ((~(all_gendered_per_entities['text'].str.contains(rf'de {proper_head}'))
                             & ~(all_gendered_per_entities['text'].str.contains(rf"d' {proper_head}"))
                             & ~(all_gendered_per_entities['text'].str.contains(rf'à {proper_head}')))
                                                    |
                             ((all_gendered_per_entities['text'].str.startswith(tuple(['ce diable de', 'ce singe de']))) & (all_gendered_per_entities['mention_len'] == 4)))
        ]

        male_count = len(proper_filtered[proper_filtered['gender'] == 'Male'])
        female_count = len(proper_filtered[proper_filtered['gender'] == 'Female'])

        male_count_list.append(male_count)
        female_count_list.append(female_count)

    to_gender_proper_singular['male_count'] = male_count_list
    to_gender_proper_singular['female_count'] = female_count_list
    to_gender_proper_singular['gendered_count'] = to_gender_proper_singular['male_count'] + to_gender_proper_singular['female_count']

    to_gender_proper_singular = to_gender_proper_singular[to_gender_proper_singular['gendered_count'] >= 1].copy()

    to_gender_proper_singular['male_ratio'] = to_gender_proper_singular['male_count'] / to_gender_proper_singular['gendered_count']
    to_gender_proper_singular['female_ratio'] = to_gender_proper_singular['female_count'] / to_gender_proper_singular['gendered_count']
    to_gender_proper_singular['max_ratio'] = np.where(to_gender_proper_singular['male_count'] > to_gender_proper_singular['female_count'], to_gender_proper_singular['male_ratio'], to_gender_proper_singular['female_ratio'])
    to_gender_proper_singular['predominant_gender'] = np.where(to_gender_proper_singular['male_count'] > to_gender_proper_singular['female_count'], 'Male', 'Female')
    to_gender_proper_singular['gender'] = np.where(to_gender_proper_singular['max_ratio'] == 1, to_gender_proper_singular['predominant_gender'], 'Ambiguous')

    male_mentions_list = to_gender_proper_singular[to_gender_proper_singular['gender'] == 'Male']["text"].tolist()
    female_mentions_list = to_gender_proper_singular[to_gender_proper_singular['gender'] == 'Female']["text"].tolist()
    ambiguous_mentions_list = to_gender_proper_singular[to_gender_proper_singular['gender'] == 'Ambiguous']["text"].tolist()

    df.loc[df[df['text'].isin(male_mentions_list)].index, 'gender'] = "Male"
    df.loc[df[df['text'].isin(female_mentions_list)].index, 'gender'] = "Female"
    df.loc[df[df['text'].isin(ambiguous_mentions_list)].index, 'gender'] = "Ambiguous"

    return df
#%%

def gender_proper_mentions_from_knowledge_base(df, insee_names_df):

    filtered_df = df[(df['prop'] == 'PROP')
                         & (df['number'] == 'Singular')
                         & (df['gender'] == 'Not_Assigned')].copy()

    original_index = filtered_df.index
    filtered_df = pd.merge(filtered_df, insee_names_df, left_on='head_word', right_on='name', how='left')
    filtered_df.index = original_index


    insee_inference_rows = filtered_df[~(filtered_df['name'].isna())
                               & (filtered_df['overall_count'] >= 20)
                               & (filtered_df['max_ratio'] >= 0.98)].copy()
    df.loc[insee_inference_rows.index.tolist(), 'gender'] = insee_inference_rows['predominant_gender'].tolist()

    # Knowledge based proper name inference ------------------------------------------------------
    male_proper_names_list = ['artagnan', 'chopin', 'athanagore', 'saint-julien', 'hitler', 'molière', 'balzac', 'homère', 'beethoven', 'mahomet', 'shakespeare', 'bonaparte', 'la fontaine', 'rousseau', 'lamartine', 'notre-seigneur', 'baudelaire', 'cicéron', 'descartes', 'stendhal', 'montaigne', 'proust', 'rabelais', 'allah', 'staline', 'goethe', 'gœthe', 'néron', 'malebranche', 'archimède', 'musset', 'schubert', 'nietzsche', 'rimbaud', 'lénine', 'sartre', 'diderot', 'kant', 'freud', 'pythagore', 'saint-germain', 'saint-léon', 'saint-françois', 'saint-georges', 'saint-vincent de paul', 'saint-mégrin', 'saint-preux', 'saint-james', 'saint-saëns', 'saint-antoine', 'saint-exupéry', "saint-thomas d' aquin", 'saint-andré', 'saint-potin', 'saint-john perse', 'saint-thomas', 'saint-vincent-de-paul', 'saint-jacques', 'saint-luc', 'saint-brice', 'saint-avit', 'saint-xist', 'saint-dieu', 'saint-nicolas', 'saint-elme', 'saint-roch', 'saint-maugon', 'saint-loup', 'saint-vincent', 'saint-hurugues', 'saint-sulpice', 'saint-rémy', 'saint-michel', 'saint-maurice', 'saint-victor', 'saint-charles', 'saint-augustin', 'saint-evremond', 'saint-guy', "saint-thomas-d'aquin", 'saint-pol', 'saint-amand', 'saint-arnaud', 'saint-benoît', 'saint-laurent', 'saint-lambert', 'saint-évremond', 'saint-vérace', 'saint-aulaire', 'saint-bernard', 'saint-philippe du roule', 'saint-hébert', 'saint-joseph', 'saint-léonard', 'saint-christophe', 'saint-cyran', 'saint-ignace', 'saint-éloi', 'saint-philippe', 'saint-séverin', 'saint-riveul', 'saint-françois de sales', 'rembrandt', 'zola', 'dostoïevski', 'ptolémée', 'télémaque', 'ronsard', 'polonius', 'béelzébuth', 'jéhovah', 'sardanapale', 'junius brutus', 'cassini', 'girodet', 'caracalla', 'humboldt', 'montesquieu', 'stendalh', 'balzac', 'potemkin', 'de balzac', 'vespasien', 'brahms', 'ïsaïe', 'verlaine', 'jacob'
]
    female_proper_names_list = ['sainte-maline', 'sainte-beuve', 'sainte-anne', 'sainte-clotilde', 'sainte-cécile', 'sainte-agnès', 'sainte-geneviève', 'sainte-marthe', "sainte-anne d' auray", 'sainte-hildegarde', 'sainte-sophie', 'sainte-brigitte', 'sainte-vierge', 'sainte-thérèse', 'sainte-périne', 'sainte-marguerite', 'sainte-odile', "sainte-claire d' ennery", 'sainte-sévère', 'sainte-aldegonde', 'sainte-austreberthe', 'sainte-foix', 'sainte-gudule', 'sainte-hermangilde', 'sainte-hermine', 'sainte-jeanne', 'sainte-misère', 'sainte-pélagie', 'sainte-justice', 'sainte-marie-des-anges', 'sainte-odile', 'sainte-orberose', 'sainte-reine', 'sainte-isabelle', 'sainte-claire-deville', 'sainte-helene', 'sainte-cécile', 'sainte-engence', 'sainte-suzanne', 'sainte-camille', 'sainte-agathe', 'sainte-amalberge', 'sainte-table', 'sainte-sophie la gigantesque', 'sainte-sidonie', 'sainte-sabine', 'sainte-routine', 'sainte-roure', 'sainte-rosalie', 'sainte-pétronille', 'sainte-ouverte', 'sainte-opportune', 'sainte-thérèse de lisieux', 'sainte-thérése', 'saintemarie', 'sainteluce', 'sainte-élisabeth', 'sainte-vierge marie', 'sainte-victoire', 'sainte-vehme', 'sainte-valère', "sainte-ursule d' uria", 'sainte-glotilde', 'sainte-germaine', 'sainte-geneviève-des-ardents', 'sainte-foire', 'sainte-foi', 'sainte-euverte', 'sainte-elisabeth de bavière', 'sainte-elisabeth', 'sainte-colombe', 'sainte-colette', 'sainte-clodilde', 'sainte-claudine', 'sainte-guillotine', 'sainte-nitouche', "sainte-n'y-touche", 'sainte-moline', 'sainte-menehould et braux - sainte - cubière', 'sainte-mary', 'sainte-marthe des lataniers', 'sainte-marie-formose', 'cendrillon', 'penthésilée', 'pomone', 'galathée', 'jj. rousseau', 'george sand']

    male_knowledge_based_rows = filtered_df[(filtered_df['text'].isin(male_proper_names_list))]
    df.loc[male_knowledge_based_rows.index, 'gender'] = 'Male'
    female_knowledge_based_rows = filtered_df[(filtered_df['text'].isin(female_proper_names_list))]
    df.loc[female_knowledge_based_rows.index, 'gender'] = 'Female'

    return df
#%%
def assign_gender_to_PER_entities(PER_entities_df, insee_path='/home/antoine/Documents/propp_modular_development/propp_fr/src/propp_fr/data/insee_names_fr_1900_2023.csv'):
    insee_names_df = pd.read_csv(insee_path,
                                 sep='\t', quoting=csv.QUOTE_NONE, low_memory=False)

    PER_entities_df['gender'] = 'Not_Assigned'
    PER_entities_df = clean_column(PER_entities_df, 'text')
    PER_entities_df = clean_column(PER_entities_df, 'head_word')

    PER_entities_df = PER_entities_df[['prop', 'cat', 'text', 'head_word', 'number', 'gender', 'mention_len']]
    grouped_PER_entities_df = PER_entities_df.groupby(['prop', 'cat', 'text', 'head_word', 'number', 'gender', 'mention_len']).first().reset_index()

    grouped_PER_entities_df = exact_ngram_match(grouped_PER_entities_df)
    grouped_PER_entities_df = always_gender_head(grouped_PER_entities_df)
    grouped_PER_entities_df = singular_ambiguous_gender_head(grouped_PER_entities_df)
    grouped_PER_entities_df = assign_gender_to_plural_head(grouped_PER_entities_df)
    grouped_PER_entities_df = assign_gender_to_singular_proper_mentions(grouped_PER_entities_df)
    grouped_PER_entities_df = assign_gender_to_complex_proper_mentions(grouped_PER_entities_df)
    grouped_PER_entities_df = assign_gender_to_proper_mentions_from_gendered_mentions(grouped_PER_entities_df)
    grouped_PER_entities_df = gender_proper_mentions_from_knowledge_base(grouped_PER_entities_df, insee_names_df)

    original_index = PER_entities_df.index
    PER_entities_df = pd.merge(PER_entities_df.drop(columns=['gender']), grouped_PER_entities_df, on=['prop', 'cat', 'text', 'head_word', 'number', 'mention_len'], how='left')
    PER_entities_df.index = original_index
    return PER_entities_df

