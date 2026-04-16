# Kinship Domain: Canon Term Lists Report

## Typological framing

English, Russian, and Spanish all belong to Murdock's (1949, Ch. 7) **Eskimo-type** kinship system. The defining feature: lineal kin (parents, children) are terminologically separated from collateral kin (aunts/uncles, cousins), and cross-cousins are not distinguished from parallel cousins. This is the most common system in Europe and the Americas, and it means we should *not* expect dramatic topological divergence in the kinship domain the way we might for color (Russian goluboj/sinij) or emotion (toska, duende). The interest here is in **granular lexicalization differences** within a shared structural type.

## What varies and why it matters

### In-law asymmetry: Russian's spouse-side distinction

The most striking cross-linguistic difference is in affinal (in-law) terminology. English has one term per structural position: "father-in-law" covers both husband's father and wife's father. Spanish is the same (suegro/suegra are gender-of-spouse neutral). Russian, however, lexicalizes the distinction: **svjokor** (husband's father) vs. **test'** (wife's father); **svekrov'** (husband's mother) vs. **tjoshcha** (wife's mother). This extends to spouse's siblings: **dever'** (husband's brother) vs. **shurin** (wife's brother); **zolovka** (husband's sister) vs. **svoyachenitsa** (wife's sister). Friedrich (1964) documents this system in detail, tracing it to the patrilocal household structure of pre-industrial Russia where a bride's relationships to her husband's family were socially and practically distinct from a groom's relationships to his wife's family.

This gives Russian roughly twice as many affinal terms as English or Spanish for the same set of relationships. If persistent homology detects topological structure in attention graphs, the Russian in-law cluster should be more finely differentiated — more distinct landmarks in the persistence diagram.

### Cousin gender: Spanish and Russian vs. English

English "cousin" is gender-neutral. Spanish splits it: primo (male) / prima (female). Russian uses a multi-word construction: dvoyurodnyj brat (literally "second-degree brother") / dvoyurodnaya sestra ("second-degree sister"). The structural move is the same in Spanish and Russian (gender the cousin), but the morphological strategy differs (separate lexeme vs. compositional modifier). This is a case where the typological pattern matches but the surface realization diverges — interesting for distributional semantics.

### Spanish-specific extended affinal terms

Spanish lexicalizes two relationships English and Russian leave to description: **consuegro/consuegra** (one's child's parent-in-law) and **concunado/concunada** (spouse of one's spouse's sibling). These are attested in the RAE dictionary and are not archaic. They represent an extension of the affinal domain into a third degree that English reaches only with circumlocution. Whether mBERT's Spanish training data contains enough instances to produce stable attention patterns is an empirical question, but including them lets us test it.

### Russian snokha/nevestka split

Russian has two terms for "son's wife": **snokha** (traditionally, from the father-in-law's perspective) and **nevestka** (from the mother-in-law's perspective, and also used for brother's wife). Friedrich (1964) treats this as a perspective-dependent distinction grounded in household authority structures. In modern usage the distinction is weakening, but both terms remain in active use and should produce distributional signal.

## Term counts

| Language | Terms |
|----------|-------|
| English  | 27    |
| Russian  | 34    |
| Spanish  | 32    |

Russian's larger inventory is almost entirely due to the affinal distinctions. The English list is smaller because English collapses these into polysemous compound terms.

## Key deviations from sources

1. **Step-relations included** despite not being Murdock's focus — needed for cross-linguistic balance and present as monomorphemic lexemes in all three languages.
2. **Russian kuzen/kuzina omitted** in favor of dvoyurodnyj brat/sestra — the French loanwords are archaic/literary and would not reflect typical mBERT training distributions.
3. **Russian yatrov' omitted** — Friedrich lists it but it is archaic/dialectal with near-zero modern corpus frequency.
4. **Great-grandparent terms omitted** in all three languages — compositional, low-frequency, and typologically uninteresting.
5. **Spanish esposo/esposa chosen** over marido/mujer for register neutrality.

## Sources

- Murdock, George Peter. 1949. *Social Structure.* Macmillan. (Anchor: kinship typology, Eskimo system classification.)
- Greenberg, Joseph H. 1966. *Language Universals, with Special Reference to Feature Hierarchies.* Mouton. (Anchor: kinship universals, markedness.)
- Friedrich, Paul. 1964. "Semantic Structure and Social Structure: An Instance from Russian." In Goodenough (ed.), *Explorations in Cultural Anthropology.* McGraw-Hill. (Anchor: Russian affinal terminology.)
- Kronenfeld, David B. 1996. *Plastic Glasses and Church Fathers.* Oxford University Press. (Augmentation: semantic extension in kinship.)
- Real Academia Espanola. 2014. *Diccionario de la lengua espanola.* 23rd ed. (Augmentation: confirms consuegro, concunado.)
