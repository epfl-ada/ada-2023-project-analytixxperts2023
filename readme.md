# Diversity in Cinema: The Influence of Diverse Casting on a Film's Performance

The website for the project's data story can be found [here](https://hmorchid.github.io/ada-project-analytixxperts2023/).

## Abstract 

The topic of diversity gained increasing attention in recent times. Many studies, like [this McKinsey & Company report](https://www.mckinsey.com/capabilities/people-and-organizational-performance/our-insights/why-diversity-matters) or [this research](https://journals.sagepub.com/doi/abs/10.1177/0146167208328062), aimed to understand why it leads to improved research, financial outcomes, and performance. We aimed to assess whether this phenomenon was also applicable to the movie industry and what usage this domain made of diversity, which was defined in this work by the diversity in the ethnicity and gender of actors.

Through this analysis, our project sought to provide a comprehensive perspective on the significance of diversity in the world of cinema and its potential implications.

## Research Questions 

We aimed to assess the impact of diversity on film's commercial performance by answering the following questions:
- How did casting ethnic and gender diversity influence the box office earnings and a film’s rating?
- How was this relationship depicted in movies with similar genres?
- How did this relationship evolve through different time periods?

## Datasets 

Our analysis was mostly performed on the [CMU movie summary corpus](https://www.cs.cmu.edu/~ark/personas/). This rich dataset regrouped information about movies, actors, the characters they played in their roles, some summaries, and even already performed NLP tasks that allowed assigning personas to the characters in the dataset. 

To supplement it, we enhanced the dataset with external information coming from the TMDB dataset, such as average ratings and vote counts for the movies that had this information available and that we were able to extract. This allowed us to judge performance on more aspects than just the box office revenue available in the CMU dataset. We also extracted some movie budgets but determined that the data wasn't sufficient and reliable enough to be used.

## Methods 

The key variables in our project were "diversity" and "performance," and we chose to measure them as follows:
- Casting diversity: Ethnicity, Gender
- Movie performance: Average ratings, Box office revenue

From an intuitive viewpoint, we considered a movie's cast to be diverse if there was a good representation of most ethnicities and genders, in balanced proportions. To quantify this diversity measure, we chose to use the [Simpson Diversity Index](https://stats.stackexchange.com/a/62744):

$D = 1 - \sum_{k=1}^{K} \left(\frac{n_k}{N}\right)^2$

Where : 
- **N** represents the total number of units in the population
- **K** denotes the different types within it.
- For each type **k**, **$n_k$** is the number of units.
- The value of **D** varies between 0 and 1, with a lower value indicating less diversity in the population.

Then, we independently compared each diversity variable with each performance variable to assess the relationships between our performance and diversity features. To answer our research questions, we performed our analysis in three different ways:
- A naïve analysis with no prior knowledge considered.
- A global causal analysis that got rid of the effect of potential confounders.
- Two more fine-grained analyses considering movies with similar genres and movies split into ten different time frames.

The reason behind the causal analysis was that we identified some potential confounders in the dataset, such as the number of languages the movies were translated into, the release year, and the movie genres. We thought that these aspects might influence both the outputs and inputs of our analysis and bring bias to our observations. In order to keep enough samples for a meaningful analysis, the matching based on genre was determined using the following formula:


$\frac{len(G_1 \cap G_2)}{max(len(G_1),len(G_2))} \geqslant t$

Where :
- $G_i$ the genre set of movie
- $i$ and $t$ a chosen threshold.

For the analysis through years, we split the dataset ranked by release year into ten sets of equal size. We then performed the same causal analysis without considering the years as confounders.

## Encountered Issues

We encountered various issues that forced us to modify our strategy:

1) We initially considered age as a type of diversity. However, since age is a continuous value, it required a different measurement tool than the Simpson Diversity Index. We opted for two approaches:
   a) Use the standard deviation of the actors' ages. However, due to the imbalance in the number of actors in films, the standard deviation was very high for films with few actors and only significant for those with a certain number of actors. The lack of actor data in the CMU database was a challenge. We tried to fetch actors' ages through other APIs, but they provided, at best, the current age of the actor, not their age at the time of filming. We could have subtracted to find the age during the film, but the data wasn't rich enough.
   b) We attempted to categorize actors' ages into age brackets to apply the Simpson Diversity Index. However, even with categorized ages, there is an inherent order (a teenager is different from an adult but closer to one than to an elderly person), which the Simpson Diversity Index does not account for.

Consequently, we abandoned this approach and decided to focus solely on ethnic and gender diversity.

2) Initially, we planned to analyze the stereotypical roles assigned to certain ethnicities and genders to deepen our research. This would have allowed us to understand if the diversity within a cast stemmed from specific personas. However, after conducting several regressions and statistical analyses, we concluded that our database of 300 personas was insufficient. Although we found a slight correlation between an actor's ethnicity and gender and the type of character portrayed, the results fluctuated too much, even with cross-validation.

Therefore, we abandoned this approach, which was, in any case, only an extension of the main study.

3) We had analyzed the influence of each measure of cast diversity on film performance metrics one by one. However, we also intended to analyze the influence of multiple measures simultaneously. Given the high number of combinations and the results from the individual measures, we judged that it probably wasn't relevant to proceed further.

## Organization within the Team

Member | Task
:------|:-----
Ilias  | Preprocessing of the dataset: standardization, metrics computation, etc.
Youssef| Naïve analysis
David  | Global analysis
Louis  | Fine-grained analyses
Hamza  | Setting up the website: interactive plots, data story, etc.

All members of the group involved themselves in giving feedback to improve each part.

## Reproducibility

Our study is divided into three files :

- `milestone2.ipynb`: Contains initial analyses and dataset generation.
- `milestone3.ipynb`: Consists of the final project.
- `src/helpers.py`: Consists of all useful functions, fully documented with references if applicable.

Both notebooks are fully documented with explanatory markdown cells. They have been fully executed, so there is no need to re-run the code. However, if someone wishes to do so, they must first run `milestone2.ipynb`, which will generate all necessary datasets in the `generated` folder.
