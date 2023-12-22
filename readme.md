# Diversity in Cinema: The Influence of Diverse Casting on a Film's Performance

[website link](https://hmorchid.github.io/ada-project-analytixxperts2023/)

## Abstract 

The topic of diversity is gaining increasing attention nowadays. Many studies, like [this McKinsey & Company report](https://www.mckinsey.com/capabilities/people-and-organizational-performance/our-insights/why-diversity-matters
) or [this research](https://journals.sagepub.com/doi/abs/10.1177/0146167208328062), aim to understand why it leads to improved research, financial outcomes, and performance. We would like to assess whether this phenomenon is also applicable to the movie industry and what usage this domain makes of diversity, which will be defined in this work by the ethnicity, age, and gender of actors.
A primary analysis will consist in identifying whether the roles assigned to diverse actors fulfil are stereotypical or hold significance to the plot.
Following this analysis, we will try to evaluate the impact of the casting's diversity on movies' performances.

Through this analysis, our project seeks to provide a comprehensive perspective on the significance of diversity in the world of cinema and its potential implications.

## Research questions 

We aim to assess the impact of diversity on film's commercial performance by answering to the following questions:
- How does casting diversity influence box office earnings and a film’s rating?
- How is this relationship depicted in movies with similar genres.
- How did this relationship evolve through different time periods?

## Datasets 

Our analysis will be mostly performed on the _"CMU movie summary corpus"_. This very rich dataset regroups information about movies, actors, the characters they play in their roles, some summaries and even already performed NLP tasks that allowed to assign personas to the characters in the dataset. 

To supplement it, we enhance the dataset with external information coming from the TMDB dataset, such as average ratings and vote counts for the movies that have this information available and that we were able to extract. This will allow us to judge performance on more aspects than the box office revenue that is available in the CMU dataset. We also extracted some movie budgets but we determined that the data wasn't sufficient and sane enough to be used.

## Methods 

The key variables in our project are "diversity" and "performance," and we have chosen to measure them as follows:
- Casting diversity: ethnicity, gender
- Movie performance: average ratings, box office revenue

From an intuitive viewpoint, we considered a movies' cast to be diverse if there is a good representation of most ethnicities and genders. To quantify this diversity measure, we chose to use the simpson diversity index:

$D = 1 - \sum_{k=1}^{K} \left(\frac{n_k}{N}\right)^2$
Here, **N** represents the total number of units in the population, and **K** denotes the different types within it. For each type **k**, **$n_k$** is the number of units. The value of **D** varies between 0 and 1, with a lower value indicating less diversity in the population.

Then, we compared independently each diversity variable with each performance variable to assess the relationships between our performance and diversity features. To answer our research questions, we performed our analysis in three different ways:
- A naïve analysis with no prior knowledge considered .
- A global causal analysis that get rid of the effect of potential confounders.
- Two more fine grained analyses considering movies with similar genres and movies split in ten different time frames.

The reason behind the causal analysis is that we identified some potential confounders in the dataset such as the number of languages the movies are translated in, the release year and the movies genres. We think that those aspects may influence both the outputs and inputs of our analysis and bring bias to our observations. In order to keep enough samples for a meaningful analysis, the matching based on genre is determined using this formula:

$\frac{len(G_1 \cap G_2)}{max(len(G_1),len(G_2))} \geqslant t$

With $G_i$ the genre set of movie i and $t$ a chosen threshold.

For the analysis through years, we split the dataset ranked by realease year in 10 sets with equal size. We then performed the same causal analysis without considering the years as confounders.


## Limitations and challenges 

Our project aims to determine how diversity within a film impacts the film's performance. However, this performance, which we measure through the film's success, can be attributed to two factors:

* Improved casting performance, thereby confirming our initial thesis.
* A broader audience that identifies with the ethnicities depicted in the film.

Therefore, not only will it be necessary to establish the correlation between a film and its success, but a significant challenge will be to determine the causal effect behind this potential correlation. Movies are subject to a lot of variability in terms of production and our analysis will be subject to strong biases. It will be very important to set up a causal analysis that rules out most of this bias as well as keeping enough data for it to be representative. Every step of the analysis will require a meticulous choice and handling of the data we will process. In addition to that, the information available for movies is still limited and we were not able to consider some important features such as movie budgets and inflation rate, even though we tried to mitigate the effect of the latter by matching movies by release year.

# Proposed timeline

Date | Task
:---:|:---:
17/11/2023|Submit the project milestone P2
20/11/2023|Start the homework H2
26/11/2023|Complete the analysis of the role dynamics
27/11/2023|Group meeting to share H2 results
01/12/2023|Submit the homework H2
02/12/2023|Complete the analysis of film performance
12/12/2023|Group meeting to review and visualize results
15/12/2023|Start creating the website
20/12/2023|Group meeting to complete data history
22/12/2023|Submit the project milestone P3

# Organization within the team

Member | Task
:----|:----
Ilias|Pre processing of the dataset: standardization, metrics computation, ...
Youssef|Naive analysis
David|Global analysis
Louis|Fine grained analyses
Hamza|Setting up the website: interactive plots, datastory, ...

All members of the group involved themselves in giving feedback to improve each part.