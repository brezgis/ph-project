# Color domain: canon term list report

## Anchor framework

The inventories are grounded in Berlin & Kay (1969), *Basic Color Terms: Their Universality and Evolution*. Their central claim is that basic color term systems evolve along a constrained trajectory, reaching a maximum of eleven terms at Stage VII: black, white, red, green, yellow, blue, brown, purple, pink, orange, gray. English instantiates this endpoint cleanly. The revised evolutionary model in Kay & Maffi (1999) updates the stage sequence but preserves the eleven-term ceiling.

## The Russian blue split

Russian is the theoretically interesting case. Where English has one basic blue term, Russian obligatorily distinguishes **синий** (sinij, dark blue) and **голубой** (goluboj, light blue). This is not a stylistic or optional distinction — it is grammatically unmarked, monolexemic, psychologically salient, and acquired early by children, meeting all of Berlin & Kay's basicness criteria.

Paramei (2005) makes the strongest case for голубой as a genuine 12th basic color term, arguing it is "culturally basic" — its salience is reinforced by deep symbolic and aesthetic associations in Russian culture (the word carries connotations of purity, sky, and idealism that синий does not). Winawer et al. (2007) provide the perceptual evidence: Russian speakers discriminate colors faster when they straddle the синий/голубой boundary than when both fall within the same category, and this advantage is eliminated by verbal (but not spatial) interference. This confirms the boundary is linguistically mediated, not just a naming preference.

Davidoff, Davies, & Roberson (1999), working with the Berinmo of Papua New Guinea, established that color categorical perception tracks linguistic boundaries generally — the Russian blue split is the most prominent instance in a major world language.

For this study, Russian gets **12 terms**. The синий/голубой split is the primary cross-linguistic contrast we expect persistent homology to detect: if mBERT's attention topology in the blue region differs structurally between Russian and English, that would be direct evidence that the model has internalized a linguistically-mediated categorical boundary.

## Spanish

Spanish maps onto the Stage VII eleven-term system without structural deviations. The main complexity is dialectal: Castilian uses *marrón* for brown where Mexican Spanish uses *café*; Castilian and Mexican Spanish use *morado* for purple where Uruguayan Spanish prefers *violeta*; *rosa* is the basic pink term across dialects though *rosado* appears adjectivally. Lillo et al. (2018) document these patterns across three dialects and confirm that despite lexical variation, the underlying color volumes are remarkably stable.

We canonicalize on Castilian forms (*marrón*, *morado*, *rosa*) because mBERT's Spanish training data (primarily Wikipedia) skews toward European Spanish. This is a pragmatic choice, not a theoretical one — we note the variants in each term entry.

Notably, Spanish does *not* split the blue region. *Azul celeste* (sky blue) exists as a compound but fails the monolexemic basicness criterion. This makes Spanish a useful control: it patterns with English in the blue region, so any topological difference between Russian and the other two languages in that region is attributable to the sinij/goluboj split rather than to some general cross-linguistic noise.

## Term counts

| Language | Terms | Notes |
|----------|-------|-------|
| English  | 11    | Standard Stage VII |
| Russian  | 12    | +голубой (light blue) |
| Spanish  | 11    | Castilian canonical forms |

## Sources

- Berlin, B. & Kay, P. (1969). *Basic Color Terms.* UC Press.
- Davidoff, J., Davies, I., & Roberson, D. (1999). Colour categories in a stone-age tribe. *Nature* 398, 203--204.
- Kay, P. & Maffi, L. (1999). Color appearance and the emergence and evolution of basic color lexicons. *American Anthropologist* 101(4), 743--760.
- Lillo, J., Moreira, H., Prado-Leon, L.R., & Melnikova, A. (2018). Basic color terms and categories in three dialects of the Spanish language. *Frontiers in Psychology* 9:761.
- Paramei, G.V. (2005). Singing the Russian blues: An argument for culturally basic color terms. *Cross-Cultural Research* 39(1), 10--38.
- Winawer, J. et al. (2007). Russian blues reveal effects of language on color discrimination. *PNAS* 104(19), 7780--7785.
