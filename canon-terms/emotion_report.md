# Emotion Domain: Canon Term List Report

## Anchor framework

The emotion inventories use a **mixed anchor**: Ekman's (1992) six basic emotions provide the cross-linguistic comparability spine (anger, disgust, fear, happiness, sadness, surprise appear in all three languages), while Wierzbicka's NSM-grounded analyses (1992, 1999) supply the culturally-specific terms that make this study worth doing. The Ekman basics ensure we can ask "do the same categories produce different topological signatures across languages?" The Wierzbicka additions let us ask the harder question: "what happens to attention topology when a language lexicalizes an emotion concept that other languages lack?"

## The culture-specific terms

### тоска (Russian)

Wierzbicka's centerpiece. She defines тоска as "how one feels when one wants some things to happen and knows that they cannot happen" (1999, Ch. 2, pp. 38-45). It is not sadness, not longing, not nostalgia — it shares features with all three but collapses none of them. English has no single lexeme for it; the best approximations ("yearning," "melancholy," "anguish") each miss something. Wierzbicka argues that тоска reflects a Russian cultural script in which passive suffering has positive value — грусть (the lighter, more reflective Russian sadness) shows the same pattern. The list includes both грусть and печаль because Wierzbicka demonstrates (Ch. 2) that Russian lexicalizes two distinct sadness concepts that English collapses into one. This is exactly the kind of structural difference persistent homology should detect.

Other Russian-specific inclusions: жалость (compassionate pity, warmer than English "pity" — Wierzbicka 1999), совесть (conscience experienced as emotion rather than moral faculty — Wierzbicka 1992), and тревога (existential anxiety, closer to German *Angst* than English "worry").

### duende (Spanish)

García Lorca's 1933 Buenos Aires lecture "Juego y teoría del duende" is the classical reference. Lorca contrasts duende with the Angel and the Muse: where the Angel gives light and the Muse gives form, duende is a dark, telluric force tied to awareness of death. Maurer (2004) isolates four elements: irrationality, earthiness, heightened death-awareness, and the diabolical.

duende is not an emotion in the Ekman sense — it is closer to an aesthetic mode of being. Its inclusion is justified precisely because it tests whether mBERT's attention patterns encode something for a culturally-specific affective concept that has no equivalent in the comparison languages. If duende shows a distinct topological signature in Spanish attention graphs, that is evidence for the project's hypothesis.

Other Spanish-specific inclusions: cariño (affectionate warmth, broader than "love" — used for family, friends, objects), añoranza (homesickness lexicalized as a single word, paralleling Russian тоска по родине), and ilusión (hopeful excitement with no English equivalent; Goddard & Wierzbicka 2002).

## Term counts

| Language | Total terms | Ekman-equivalent | Culture-specific |
|----------|------------|------------------|-----------------|
| English  | 18         | 7                | 11              |
| Russian  | 19         | 7                | 12              |
| Spanish  | 22         | 7                | 15              |

English has fewer culture-specific terms because it *is* the baseline — Ekman's model was developed on English speakers. The "culture-specific" English terms (love, shame, guilt, pride, jealousy, envy, hope, pity, longing, anxiety, joy) are included not as deviations but as comparison targets for Russian and Spanish terms that partially overlap them.

## Sources

| Source | Role |
|--------|------|
| Ekman 1992. "An argument for basic emotions." *Cognition and Emotion* 6(3-4): 169-200. | Comparability spine |
| Wierzbicka 1999. *Emotions across Languages and Cultures.* Cambridge. | Primary cross-linguistic analysis |
| Wierzbicka 1992. *Semantics, Culture, and Cognition.* Oxford. | NSM framework; Russian душа, совесть |
| Apresjan, V. & J. Apresjan 2000. "Metaphor in the semantic representation of emotions." In *Systematic Lexicography*, Oxford. | Russian emotion metaphor |
| Pavlenko 2005. *Emotions and Multilingualism.* Cambridge. | Russian-English bilingual emotion concepts |
| García Lorca 1933. "Juego y teoría del duende." Lecture, Buenos Aires. | duende as aesthetic-emotional concept |
| Maurer (ed.) 2004. *In Search of Duende.* New Directions. | Scholarly framing of Lorca |
| Goddard & Wierzbicka (eds) 2002. *Meaning and Universal Grammar.* Benjamins. | NSM applied to Spanish; ilusión |

## Sources I wanted but could not fully verify

- **Pavlenko on тоска specifically**: Pavlenko 2005 discusses Russian-English emotion differences broadly, but I could not confirm she treats тоска as a named case study at a specific page. I cite her as augmentation, not anchor.
- **Apresjan page numbers**: The chapter on emotion metaphor is Ch. 7 of *Systematic Lexicography* (2000), originally in *Voprosy iazykoznaniia* 3 (1993). I could not confirm exact page range in the 2000 edition.
- **Goddard & Wierzbicka 2002 on Spanish ilusión**: I cite this based on the NSM framework's coverage of Spanish, but could not confirm the exact chapter or page where ilusión is analyzed. The claim that ilusión is a Spanish-specific emotion concept is well-established in the cross-linguistic literature, but the specific Goddard & Wierzbicka citation may need verification against Anna's copy.
