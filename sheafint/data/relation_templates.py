"""
Hand-curated relation-fact templates for hypergraph experiments.

Six relations with five paraphrase templates each cover capital cities,
official languages, currencies, company CEOs, founding years, and landmark
locations. ``build_facts`` instantiates the templates with the provided
entity tables to produce a list of fact dicts ``{relation_id, paraphrases}``.

Used by the holonomy null calibration and other relation-hypergraph
experiments where the paraphrase set has known relation structure.
"""

from typing import Dict, List

RELATION_TEMPLATES: Dict[str, List[str]] = {
    'capital': [
        '{entity} is the capital of',
        'The capital of {country} is',
        '{entity}, capital of',
        'The city of {entity} serves as the capital of',
        "{country}'s capital is",
    ],
    'language': [
        'The official language of {country} is',
        'In {country}, people speak',
        "{country}'s national language is",
        'The language spoken in {country} is',
        'Most people in {country} speak',
    ],
    'currency': [
        'The currency of {country} is the',
        '{country} uses a currency called the',
        'Money in {country} is denominated in',
        "{country}'s monetary unit is the",
        'The official currency of {country} is the',
    ],
    'ceo': [
        'The CEO of {company} is',
        '{company} is led by',
        "{company}'s chief executive officer is",
        'The head of {company} is',
        'Running {company} is',
    ],
    'founded': [
        '{company} was founded in',
        'The founding year of {company} was',
        '{company} began operations in',
        '{company} traces its origins to',
        'The year {company} was established was',
    ],
    'located_in': [
        '{landmark} is located in',
        'You can find {landmark} in',
        'The {landmark} stands in',
        '{landmark} is situated in the city of',
        'Tourists visit {landmark} which is in',
    ],
}

ENTITIES: Dict[str, List[tuple]] = {
    'capital': [
        ('Paris', 'France'), ('Berlin', 'Germany'), ('Tokyo', 'Japan'),
        ('Madrid', 'Spain'), ('Rome', 'Italy'), ('Ottawa', 'Canada'),
        ('London', 'the United Kingdom'), ('Cairo', 'Egypt'),
        ('Athens', 'Greece'), ('Brasilia', 'Brazil'),
    ],
    'language': [
        ('France',), ('Germany',), ('Japan',), ('Spain',),
        ('Italy',), ('Brazil',), ('Russia',), ('China',),
        ('Egypt',), ('Greece',),
    ],
    'currency': [
        ('Japan',), ('the United Kingdom',), ('Switzerland',),
        ('Mexico',), ('India',), ('Russia',), ('China',),
        ('Brazil',), ('Egypt',), ('Turkey',),
    ],
    'ceo': [
        ('Apple',), ('Google',), ('Microsoft',), ('Amazon',),
        ('Meta',), ('Tesla',), ('Netflix',), ('Nvidia',),
        ('Adobe',), ('Salesforce',),
    ],
    'founded': [
        ('Apple',), ('Google',), ('Microsoft',), ('IBM',),
        ('Tesla',), ('Toyota',), ('Sony',), ('Samsung',),
        ('Disney',), ('Boeing',),
    ],
    'located_in': [
        ('Eiffel Tower', 'Paris'), ('Statue of Liberty', 'New York'),
        ('Big Ben', 'London'), ('Colosseum', 'Rome'),
        ('Brandenburg Gate', 'Berlin'), ('Sydney Opera House', 'Sydney'),
        ('Acropolis', 'Athens'), ('Forbidden City', 'Beijing'),
        ('Kremlin', 'Moscow'), ('Taj Mahal', 'Agra'),
    ],
}


def build_facts(
    n_relations: int = 6,
    paraphrases_per_fact: int = 5,
) -> List[dict]:
    """
    Instantiate the templates into a list of facts.

    Each fact is ``{'relation_id': str, 'paraphrases': List[str]}``. The
    first ``paraphrases_per_fact`` templates of each relation are used.
    """
    facts: List[dict] = []
    rels = list(RELATION_TEMPLATES.keys())[:n_relations]
    for rel in rels:
        templates = RELATION_TEMPLATES[rel][:paraphrases_per_fact]
        for entry in ENTITIES[rel]:
            kwargs = {'entity': entry[0]}
            if len(entry) > 1:
                kwargs['country'] = entry[1]
                kwargs['landmark'] = entry[0]
            else:
                kwargs['country'] = entry[0]
                kwargs['company'] = entry[0]
                kwargs['landmark'] = entry[0]
            paraphrases: List[str] = []
            for t in templates:
                try:
                    paraphrases.append(t.format(**kwargs))
                except KeyError:
                    paraphrases.append(t.format(
                        entity=entry[0], country=entry[0],
                        company=entry[0], landmark=entry[0],
                    ))
            facts.append({'relation_id': rel, 'paraphrases': paraphrases})
    return facts
