# Abstract 

The topic of diversity is gaining increasing attention nowadays. It is prevalent in many fields and numerous studies such as [this McKinsey & Company report](https://www.mckinsey.com/capabilities/people-and-organizational-performance/our-insights/why-diversity-matters
) or [this research](https://journals.sagepub.com/doi/abs/10.1177/0146167208328062) try to explain the reasons behind this dynamic that seems to achieve superior research, financial and performance. We would like to assess whether this phenomenon is also applicable to the movie industry and what usage this domain makes of diversity, which will be defined in this work by the age, ethnicity and gender of actors.
A primary analysis will consist in identifying whether the roles assigned to diverse actors fulfil are stereotypical or hold significance to the plot.
Following this analysis, we will try to evaluate the impact of the casting's diversity on movies' performances.

Through this analysis, our project seeks to provide a comprehensive perspective on the significance of diversity in the world of cinema and its potential implications.

# Research questions 

Analysis of the dynamics of the roles occupied by people from diverse backgrounds :
- How can roles be categorized based on the ethnicity, gender and age of the actors ?
- What is the centrality of characters portrayed by diverse actors in the plots of the movies ?
- What type of stereotypical roles are frequently assigned to these actors and how prevalent are they in films ?

Impact of diversity on film's commercial performance :
- How does casting diversity influence box office earnings, film’s rating and the likelihood of producing sequels?
- Is there a correlation between the linguistic diversity of films and their box office performance ?
- Are there any differences in terms of diversity in the casting nowadays compared to the past ?

# Datasets 

Our analysis will be mostly performed on the "CMU movie summary corpus". This very rich dataset regroups information about movies, actors, the characters they play in their roles, some summaries and even already performed NLP tasks that allowed to assign personas to the characters in the dataset. 

To supplement it, we enhance the dataset with external information coming from the TMDB dataset, such as average ratings, vote counts and budgets for the movies that have this information available and that we were able to extract. This will allow us to judge performance on more aspects than the box office revenue that is available in the CMU dataset.

# Methods 

The key variables in our project are "diversity" and "performance," and we have chosen to measure them as follows:
- Casting diversity: D = {Ethnicity, gender, age}
- Film performance/success: S = {Box office earnings, ratings, languages, number of subsequent films}

With the complete dataset, our objective is to analyze the correlation between the power set of D and the power set of S. This analysis will provide insights into which variable(s) in D may influence variable(s) in S.

Furthermore, assuming that we identify subsets D' < D and S' < S, we will introduce a third dimension T, to observe how the results evolve over time. To achieve this, we will group movies by time periods and examine whether gender or racial inclusion in the industry over time has had an impact on the success of a movie.

More precisely, in order to assess movies' performances, we will try to perform a causal analysis on the different attributes of the performance set S by making a propensity score matching. The matching will minimize the following distance between pairs:
$$Argmin_{i,j} (\alpha_k(M_{c_k,i} - M_{c_k,j}))$$
with i,j being two samples of the dataset and $M_{c_k}$ the set of features that may bias the analysis such as available language translations, duration, country, genres, etc and finally $\alpha_k$ a vector of weights assigned to each movie feature. The same method will be used to evaluate the presence of diverse casts through years.

To perform the analysis of stereotypes, we will compute a set of probabilities $P(persona|D)$ using a logistic regression and we will segment some attributes of the set D in order to restrict the number of available features and get exploitable results. We will then rank these probabilities and chose a threshold $t_s$ for which we will consider a certain type of person to be likely of being assigned to a persona and therefore answer our research question.

Finally, we will assess the importance to the plot of the characters impersonated by diverse actors by counting the presence of their character's names in the plot summaries. This will be a good indicator of the importance of their role to the plot.


# Limitations and challenges 

Our project aims to determine how diversity within a film impacts the film's performance. However, this performance, which we measure through the film's success, can be attributed to two factors:

* Improved casting performance, thereby confirming our initial thesis.
* A broader audience that identifies with the ethnicities depicted in the film.

Therefore, not only will it be necessary to establish the correlation between a film and its success, but a significant challenge will be to determine the causal effect behind this potential correlation.

Movies are subject to a lot of variability in terms of production and our analysis will be subject to a strong bias. Therefore, it will be very important to set up a causal analysis that rules out most of this bias as well as keeping enough data for it to be representative. Every step of the analysis will require a meticulous choice and handling of the data we will process.  

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

The tasks are distributed fairly such that each of us will contribute to the project. We are planning several meetings in the remaining weeks to discuss the issues we are facing and the progress of our tasks. 
