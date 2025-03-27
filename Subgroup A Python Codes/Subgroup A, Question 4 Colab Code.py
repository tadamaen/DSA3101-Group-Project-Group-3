"""
## Importing The Necessary Packages
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

"""Other Settings Implemented Using Pandas

This is an optional step but the following block of code below helps to change the output structure of the code such that

1) All the columns of the dataset will be printed

2) The width of the dataset output is not limited to the display width of Google Colab

3) Prevent wrapping the output to multiple lines on Google Colab to improve readibility
"""

# Show all columns of the dataset when printed
pd.set_option('display.max_columns', None)

# Don't limit the display width of the output
pd.set_option('display.width', None)

# Don't wrap the output to multiple lines
pd.set_option('display.expand_frame_repr', False)

"""## Load Data"""

def load_data(file_path, is_current=True):
    '''
    Args:
      file_path: path to dataset
    Returns:
      df: DataFrame containing dataset
    '''
    df = pd.read_excel(file_path)

    if is_current:
      df = df.drop(['Email Address', 'time_entry'], axis=1)

    return df

df = load_data('uss_survey_responses.xlsx')

"""## Business Question 4: Impact Of Marketing Strategies On Guest Behaviour

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#### Overview Of Using Marketing Strategies:

To assess the impact of marketing strategies on guest behavior, it is essential to first identify the characteristics of customers who spend at Universal Studios Singapore (USS). Tailoring marketing efforts to underrepresented groups without considering the broader customer base may limit their effectiveness in enhancing the overall guest experience. This, in turn, could result in minimal or no significant increase in demand for USS, potentially restricting revenue growth and preventing the park from maximizing its financial performance.

However, merely identifying customer characteristics is not sufficient to develop effective marketing strategies. It is equally important to analyze shifts in customer trends over time—for example, observing a consistent increase in the percentage of visitors aged 13 to 21. By tracking these evolving patterns, USS can better predict future customer demographics and tailor its marketing efforts accordingly.

Focusing on growing customer segments allows USS to implement targeted strategies that align with emerging trends, maximizing demand and ensuring sustained business growth. By comparing past demographic data with current visitor profiles, we can assess percentage changes across different customer segments. This analysis enables USS to better understand evolving visitor trends and develop targeted strategies to align with shifting consumer behaviors.

We need to analyze past campaign data to study changes in guest segment size and satisfaction. To identify trends and analyze shifts in customer patterns, it is essential to utilize historical demographic data from visitors to Universal Studios Singapore (USS), ideally from 10 or more years ago. This long-term dataset provides a more accurate basis for trend analysis, minimizing short-term fluctuations and errors.

To tackle this business question, we need to

Step 1: From the current survey dataset, we will obtain and analyze the demographics of visitors, including age range, visitor type, gender, salary range as well as time of visitations and special occasions

Step 2: Similar to Step 1, we will obtain and analyze the the demographics of visitors who visited USS some time ago

Step 3: We will then compare the demographics of current visitors versus visitors several years ago using some visualizations + identify some noticeable increasing trends, as well as the majority classes

Step 4: We will implement effective strategies to tailor to these large groups of people to increase user experience and ultimately increase revenue/profits for USS

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

### Step 1: Obtain And Analyze The Demographics Of Current Visitors In USS

From the current survey dataset that we have obtained, we can categorize the guest types based on several factors listed below:

*   Visitor Demographics - Age, Gender, Nationality (Tourist/Local)
*   Visitor Type - Solo Travellor/Visiting With Friends/Family/Children etc.
*   Time Of Day - Weekdays/Weekends/Public Holidays etc.
*   Special Events - Halloween/Christmas etc.

We will explore each bullet point one by one. We will first start with the Visitor Demographics.

--------------------------------------------------------------------------------

#### Visitor Demographics

For the point on Visitor Demographics, there are several features that we can explore. These include:

*   **Age Of Visitors**
*   **Gender Of Visitors**
*   **Nationality Of Visitors (Tourists/Local)**

###### Age Of Visitors

First, for the age of visitors, we split the visitors ages into various categories, such as

1.   Below 12 years old (Children)
1.   13 to 20 years old (Teenagers)
3.   21 to 34 years old (Young Adults)
4.   35 to 49 years old (Mid-Career Adults)
5.   50 to 64 years old (Mature Adults)
6.   65 years old and above (Seniors)

We can plot the number of each type of visitors using a bar chart, categrorized by age group. The column that we will be focusing on is `q2_1` (What is your age range?)
"""

# Get count by age cateogry
def get_age_count(df, is_current=True):
  '''
  Args:
    df: DataFrame containing dataset
    is_current: Boolean indicating if data is current survey responses (True)
                or historical reviews (False)
  Returns:
    age_count: Series containing count of each age category
  '''

  # Specify order
  order = ["Below 12 Years Old", "13 To 20 Years Old", "21 To 34 Years Old",
         "35 To 49 Years Old", "50 To 64 Years Old", "65 Years Old And Above"]

  if is_current:
    age_count = df['q2_1'].value_counts().reindex(order, fill_value=0)
  else:
    age_count = df['age_range'].value_counts().reindex(order, fill_value=0)

  age_count.index = pd.Categorical(age_count.index, categories=order, ordered=True)

  return age_count

# Plot bar chart
def plot_bar(data, title, xlabel, ylabel, color='skyblue', rotation=30):
  '''
  Args:
    data: Series containing data to be plotted
    title: Plot title
    xlabel: x-axis label
    ylabel: y-axis label
    color: Bar color
    rotation: x-tick label rotation
  '''
  plt.figure(figsize=(10, 6))
  bars = plt.bar(range(len(data)), data.values, color=color)

  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.xticks(range(len(data)), data.index, rotation=rotation, ha='right')
  plt.grid(axis='y', linestyle='--', alpha=0.3)

  # Add count labels
  for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{int(height)}',
                 ha='center', va='bottom')

  plt.tight_layout()
  plt.show()

"""--------------------------------------------------------------------------------

###### Gender Of Visitors

We can plot the proportion of gender of visitors using a pie chart. The column that we will be focusing on is `q2_2` (What is your gender?)

NOTE: For simplicity, we only assume two possible genders - Male and Female. We have made this question a mandatory question for the survey respondants to answer and give them only two options to choose from. We have explained to the respondants on the main page that their information will be kept confidential and will not be exposed to the public or on any websites. The information will only be used for data analysis and insights.
"""

# Get count by gender
def get_gender_count(df, is_current=True):
  '''
  Args:
    df: DataFrame containing dataset
    is_current: Boolean indicating if data is current survey responses (True)
                or historical reviews (False)
  Returns:
    gender_count: Series containing count of each gender
  '''

  if is_current:
    gender_count = df['q2_2'].value_counts()
  else:
    gender_count = df['gender'].value_counts()

  return gender_count

# Plot pie chart
def plot_pie(data, title, colors):
  plt.figure(figsize=(9, 9))
  plt.pie(data, labels=data.index, autopct='%1.1f%%',
          colors=colors, startangle=90, wedgeprops={'edgecolor': 'black'})
  plt.title(title)
  plt.legend(title=data.name, loc="best")
  plt.tight_layout()
  plt.show()

"""--------------------------------------------------------------------------------

###### Nationality Of Visitors

We can similarly plot the proportion of Nationality of visitors using a pie chart. The column that we will be focusing on is `q3` (Are you a tourist or a local?)

NOTE: For simplicity, we only assume two possible answers - tourist or local. This is because if we ask the tourists further on which country that they are from, there will be a myriad of different answers as there are over 200 countries in the world. This might result in too many categories for us to analyze and given the limited number of survey responses that we have collected, some countries might only have 1 or 2 responses which cannot be used to compare differences subsequently in Step 3
"""

# Get count by nationality
def get_nationality_count(df, is_current=True):
  '''
  Args:
    df: DataFrame containing dataset
    is_current: Boolean indicating if data is current survey responses (True)
                or historical reviews (False)
  Returns:
    nationality_count: Series containing count of locals and tourists
  '''

  if is_current:
    nationality_count = df['q3'].value_counts()
  else:
    nationality_count = df['nationality'].value_counts()

  return nationality_count

"""--------------------------------------------------------------------------------

#### Visitor Type

The types of visitors can be categorized as follows:

*   Solo Traveller
*   Visiting with Friends
*   Families with Young Children
*   Families with Teenagers
*   Families with Elderly

To visualize the distribution of the above visitor types, we will plot a bar chart using column `q1` of the `uss_survey_responses.xlxs` dataset. (Which type of theme park visitor best describes you?)
"""

# Get count by visitor type
def get_visitor_type_count(df, is_current=True):
  '''
  Args:
    df: DataFrame containing dataset
    is_current: Boolean indicating if data is current survey responses (True)
                or historical reviews (False)
  Returns:
    visitor_type_count: Series containing count of each visitor type
  '''
  # Specify order
  order = ['Solo Traveller', 'Visiting With Friends', 'Family With Young Children',
          'Family With Teenagers', 'Family With Elderly']

  if is_current:
    visitor_type_count = df['q1'].value_counts()
  else:
    visitor_type_count = df['visitor_type'].value_counts()

  visitor_type_count.index = pd.Categorical(visitor_type_count.index, categories=order, ordered=True)
  visitor_type_count = visitor_type_count.sort_index()

  return visitor_type_count

"""--------------------------------------------------------------------------------

#### Time Of Day

For the feature on the Time Of Day, we can break down into two categories:

**1st Category: When do visitors usually spend their day at USS?**

**2nd Category: What time of the day constitutes the highest demand?**

##### Category 1:

For the first category of "When do visitors usually spend their day at USS?", we can make use of column `q10` (When do you usually visit theme parks or attractions like USS?). This question targets the day type aspect of visitation. We can make use of a bar graph to plot the number of responses who select the possible options below.

The Possible Options Include:

*   Weekdays
*   Weekends
*   Public Holidays
*   School Holidays
*   Special Events (Halloween, Summer Festival, Christmas)
*   Other Inputs By User (Can Be Anything)



However, users are allowed to select multiple options from the list above and hence, we cannot plot the bar graph immediately. We need to perform some data cleaning and manipulation first before visualizing.

First, we need to identify all the unique categories of `q10` as some users might have inputted their preferred days under the "Other Inputs By User" section of the survey question.

We observe that there are other entries such as 'When I feel like it' and '1 day in 25 years' that are in the resulting set of unique values, but are not available in the fixed options in the survey.

Data Cleaning Process:

1) Extract the `q10` column of the dataset

2) Separate the options using the comma delimiter for entries with more than 1 option

3) Performing One-Hot Encoding to create new columns - `weekdays`, `weekends`, `public_holidays`, `school_holidays` and `special_events` and others in the set containing 0 for absence and 1 for presence

4) Count the total number of 1's for each column of the 7 new columns created
"""

# Get count of days when visitors usually visit USS
def get_occasion_count(df, is_current=True):
  '''
  Args:
    df: DataFrame containing dataset
    is_current: Boolean indicating if data is current survey responses (True)
                or historical reviews (False)
  Returns:
    occasion_count: Series containing count of each occasion that visitors visit USS
  '''

  if is_current:
    # Replace "Special Events (...)" with "Speciak Events" using regex
    col = df['q10'].replace(r'Special Events \(.*\)', 'Special Events', regex=True)

    # Separate the options using the comma delimiter for entries with more than 1 option
    split_q10 = col.str.split(',', expand=True)

    # Define the categories we want to create columns for
    categories = ['Weekdays', 'Weekends', 'Public Holidays',
                  'School Holidays', 'Special Events', 'When I feel like it',
                  '1 day in 25 years']

    # Create a new dataframe with columns for each category
    one_hot_df = pd.DataFrame(0, index = df.index, columns = categories)

    # Fill the new columns with 1s where the category is present
    for category in categories:
        one_hot_df[category] = split_q10.apply(lambda row: 1 if category in row.values else 0, axis = 1)

    # Count the total number of 1's for each new column created
    occasion_count = one_hot_df.sum()

    # Remove the "When I feel like it" and "1 day in 25 years" categories
    occasion_count = occasion_count.drop(['When I feel like it', '1 day in 25 years'])

    # Define the desired order
    order = ["Weekdays", "Weekends", "Public Holidays", "School Holidays", "Special Events"]

    # Reindex to force the order
    occasion_count = occasion_count.reindex(order)

  else:
    # Extract the day_preferred column
    dfhistory_day = df['day_preferred']

    # Define the desired order
    order = ["Weekdays", "Weekends", "Public Holidays", "School Holidays", "Special Events"]

    # Count occurrences and reindex to match the desired order
    occasion_count = dfhistory_day.value_counts().reindex(order, fill_value=0)

  return occasion_count

"""--------------------------------------------------------------------------------

##### Category 2:

For the second category of "What time of the day constitutes the highest demand?", we can make use of several columns of the dataset to answer the question. This question targets the time aspect of visitation.

The columns that we will consider include:

*   `q14_1`: At what time of the day do you usually visit roller coasters?
*   `q14_2`: At what time of the day do you usually visit water rides?
*   `q14_3`:	At what time of the day do you usually visit 3D/4D experiences?
*   `q14_4`:	At what time of the day do you usually visit performances?
*   `q14_5`:	At what time of the day do you usually visit roadshows?
*   `q14_6`:	At what time of the day do you usually visit eateries and restaurants?
*   `q14_7`:	At what time of the day do you usually visit souvenir shops?
*   `q14_8`:	At what time of the day do you usually visit other rides (carousel rides, teacup rides etc.)?


**However, these columns selected target the popular times for each attraction, which we are not focusing on for this business question. We are instead focusing on the most popular timings in USS in general (not specific to each attraction). Instead, we should modify the columns above such that we calculate the combined count of each time category for all the eight columns.**

We can make use of another bar graph to plot the number of responses who select the possible options below.

The Possible Options Include:

*   Early Morning (8am to 10am)
*   Late Morning (10am to 12pm)
*   Early Afternoon (12pm to 2pm)
*   Late Afternoon (2pm to 4pm)
*   Evening (4pm to 6pm)
*   Night (6pm to 9pm)
*   I Do Not Visit

For these questions, users are not allowed to select any other option or type other suitable timings. Hence, there are only six categories that we will consider. Also, we will exclude the I Do Not Visit entries as we are only interested in the visitation timings of USS.

NOTE: The y-axis values of the bars do not truly reflect the actual number of respondants who visited USS due to the manipulation of the column values, as well as it is not possible for a certain visitor to be in multiple attractions at one time. There might be certain inaccuracies. Instead, we will just note the trend of the bars over time from early morning to night and observe which bars are the longest/shortest.
"""

# Get count of each time slot
def get_time_slot_count(df, is_current=True):
  '''
  Args:
    df: DataFrame containing dataset
    is_current: Boolean indicating if data is current survey responses (True)
                or historical reviews (False)
  Returns:
    time_slot_count: Series containing count of each time slot
  '''
  # Specify order
  order = ["Early Morning (8am to 10am)", "Late Morning (10am to 12pm)",
         "Early Afternoon (12pm to 2pm)", "Late Afternoon (2pm to 4pm)",
         "Evening (4pm to 6pm)", "Night (6pm to 9pm)"]

  if is_current:
    # Extract the relevant columns for time ranges
    popular_time_range = df[['q14_1', 'q14_2', 'q14_3', 'q14_4', 'q14_5', 'q14_6', 'q14_7', 'q14_8']]

    # Flatten the DataFrame and split entries with commas into separate values
    flattened_time_range = popular_time_range.values.flatten()

    # Split by commas for entries with multiple options
    split_time_range = [item.strip() for sublist in flattened_time_range for item in str(sublist).split(',')]

    # Convert to a pandas Series for easier counting
    time_range_series = pd.Series(split_time_range)

    # Count the occurrences of each time slot
    time_slot_count = time_range_series.value_counts()

    # Remove invalid entries
    time_slot_count = time_slot_count[~time_slot_count.index.isin(['', 'Lunch (11am to 2pm)', 'i Do Not Visit'])]

    # Add 3 "i Do Not Visit" entries to the count of "I Do Not Visit"
    time_slot_count = time_slot_count.rename(index = {'i Do Not Visit': 'I Do Not Visit'})
    time_slot_count['I Do Not Visit'] += 3

    # Remove the "I Do Not Visit" category for plotting
    time_slot_count = time_slot_count.drop(['I Do Not Visit'])

    # Reindex the time_slot_count to ensure the desired order
    time_slot_count = time_slot_count.reindex(order, fill_value=0)

  else:
    # Extract the time_of_day column
    dfhistory_time = df['time_of_day']

    # Count occurrences and reindex to match the desired order
    time_slot_count = dfhistory_time.value_counts().reindex(order, fill_value=0)

  # Convert index to categorical with specified order
  time_slot_count.index = pd.Categorical(time_slot_count.index,
                                         categories = order, ordered = True)

  return time_slot_count

"""--------------------------------------------------------------------------------

#### Special Events

To investigate the impact of special events on visitors, we will first analyze column `q7` (What are the factors that will influence your decision to visit a theme park like USS?) to see if `Special Events` is an important factor to survey respondants.

`q7` consists of the following options, where survey respondants are allowed to choose several options:

*   Weather Conditions
*   Holiday Seasons
*   Wait Times For Rides
*   Attraction Variety
*   **Special Events**
*   Cost And Ticket Prices
*   Location And Accessibility
*   Reputation And Reviews
*   Safety And Cleanliness
*   Other inputs by respondants
"""

# Get count of each factor influencing visitors' decision to visit USS
def get_factor_count(df, is_current=True):
  '''
  Args:
    df: DataFrame containing dataset
    is_current: Boolean indicating if data is current survey responses (True)
                or historical reviews (False)
  Returns:
    factor_count: Series containing count of each factor influencing visitation
  '''
  if is_current:
    # Extract the q7 column from the dataset
    q7_column = df['q7']

    # Replace "Holiday seasons" with "Holiday Seasons" and "Weather conditions" with "Weather Conditions"
    q7_column = q7_column.replace({'Holiday seasons': 'Holiday Seasons',
                                  'Weather conditions': 'Weather Conditions'})

    # Separate the options using the comma delimiter for entries with more than 1 option
    split_q7 = q7_column.str.split(',', expand=True).apply(lambda x: x.str.strip())

    # Define the categories we want to create columns for
    categories = ['Weather Conditions', 'Holiday Seasons', 'Wait Times For Rides',
                  'Attraction Variety', 'Special Events', 'Cost And Ticket Prices',
                  'Location And Accessibility', 'Reputation And Reviews',
                  'Safety And Cleanliness', 'Whether I am working or not!',
                  'Aesthetics', 'Thrill factor (not to be confused with scare factor)']

    # Create a new dataframe with columns for each category
    one_hot_df = pd.DataFrame(0, index = df.index, columns = categories)

    # Fill the new columns with 1s where the category is present
    for category in categories:
        one_hot_df[category] = split_q7.apply(lambda row: 1 if category in row.values else 0, axis = 1)

    # Count the total number of 1's for each new column created
    factor_count = one_hot_df.sum()

    # Remove the categories with count = 1
    factor_count = factor_count.drop(['Whether I am working or not!', 'Aesthetics',
                                  'Thrill factor (not to be confused with scare factor)'])

    # Sort the factor_count in descending order
    factor_count = factor_count.sort_values(ascending = False)

  factor_count.index = pd.Categorical(factor_count.index, categories=categories, ordered=True)

  return factor_count

"""To further investigate which particular special events attract visitors, we will analyze column `q8` (What type of events influence your decision to visit?), which consists of the following options:

*   Minion Land Grand Opening
*   Halloween Horror Night
*   A Universal Christmas
*   None Of The Above

Since respondants are allowed to choose several options, we will perform One-Hot Encoding, then use a bar chart to visualize the total count for each option.
"""

# Get count of special events influencing visitation
def get_event_count(df, is_current=True):
  '''
  Args:
    df: DataFrame containing dataset
    is_current: Boolean indicating if data is current survey responses (True)
                or historical reviews (False)
  Returns:
    event_count: Series containing count of each special event influencing visitation
  '''
  if is_current:
    # Extract the q7 column from the dataset
    q8_column = df['q8']

    # Separate the options using the comma delimiter for entries with more than 1 option
    split_q8 = q8_column.str.split(',', expand=True).apply(lambda x: x.str.strip())

    # Define the categories we want to create columns for
    categories = ['Minion Land Grand Opening',
                  'Halloween Horror Night',
                  'A Universal Christmas',
                  'None Of The Above']

    # Create a new dataframe with columns for each category
    one_hot_df = pd.DataFrame(0, index = df.index, columns = categories)

    # Fill the new columns with 1s where the category is present
    for category in categories:
        one_hot_df[category] = split_q8.apply(lambda row: 1 if category in row.values else 0, axis = 1)

    # Count the total number of 1's for each new column created
    event_count = one_hot_df.sum()

    # Sort the event_count in descending order
    event_count = event_count.sort_values(ascending = False)

  return event_count

"""--------------------------------------------------------------------------------

#### Satisfaction Ratings

To analyze the guest satisfaction of USS of recent survey respondents, we will be using column `q15` (How likely are you to recommend USS to others?) of the dataset. This column consists of scores from 1-10, where a score of 1 means that respondents are not at all likely to recommend USS to others while a score of 10 means that respondents are extremely likely to recommend USS to others
"""

# Get count of each guest satisfaction score
def get_satisfaction_count(df, is_current=True):
  '''
  Args:
    df: DataFrame containing dataset
    is_current: Boolean indicating if data is current survey responses (True)
                or historical reviews (False)
  Returns:
    satisfaction_count: Series containing count of each guest satisfaction score
  '''
  if is_current:
    satisfaction_count = df['q15'].value_counts().sort_index()
  else:
    satisfaction_count = df['rating'].value_counts().sort_index()

  return satisfaction_count

"""#### Step 1 Helper Function"""

# Step 1: Obtain And Analyze The Demographics Of Current Visitors In USS
def analyse_survey_responses(df):
  '''
  Args:
    df: DataFrame of survey responses
  '''

  print('\n\033[1m' + 'Step 1: Obtain And Analyze The Demographics Of Current Visitors In USS' + '\033[0;0m\n')


  ### Analyse current visitor demographic by age range ###
  print('\n\033[1m' + 'Analysing Number of Respondents by Age Range' + '\033[0;0m\n')
  age_count_current = get_age_count(df)
  plot_bar(age_count_current,
          title='Number Of Respondents Based On Age Range (Current Data)',
          xlabel='Age Range Categories',
          ylabel='Number of Respondents')

  print("\n\033[1mKey Observations And Insights:\033[0;0m\n"
      "1) The age range category that has the highest number of respondants is 21 To 34 Years Old (Young Adults). This is followed by 13 To 20 Years Old (Teenagers) and Below 12 Years Old (Children).\n"
      "2) The number of respondants aged 34 years and below contributes to the majority of the total number of survey responses, almost 75% of the total number of respondants are in this age range.\n"
      "3) From 35 years old onwards, the bar graph shows a continuous decline in the number of respondants. The lowest number of respondants turned out to be 65 years old and above (Elderly).\n"
      "We will analyse the age range categories further using heatmaps further in Step 3 when we are interested in comparing the proportions of respondants with particular age ranges over time.\n")


  ### Analyse current visitor demographic by gender ###
  print('\n\033[1m' + 'Analysing Number of Respondents by Gender' + '\033[0;0m\n')
  gender_count_current = get_gender_count(df)
  plot_pie(gender_count_current,
          title='Proportion of Male and Female Participants (Current Data)',
          colors=['#ff69b4', '#377eb8'])

  print("\n\033[1mKey Observations And Insights:\033[0;0m\n"
        "The pie chart reveals that 61% of respondents are female, while the remaining 39% are male. This indicates that the majority of visitors to Universal Studios Singapore (USS) are female.\n"
        "However, to determine whether this trend is a recent development or a long-standing pattern, it is essential to compare these proportions with historical data.\n"
        "Analyzing past visitor demographics will help assess any shifts in gender distribution over time.\n")


  ### Analyse current visitor demographic by nationality ###
  print('\n\033[1m' + 'Analysing Number of Respondents by Nationality' + '\033[0;0m\n')
  nationality_count_current = get_nationality_count(df)
  plot_pie(nationality_count_current,
          title='Proportion of Locals and Tourists (Current Data)',
          colors=['#AEC6CF', '#FA8072'])

  print("\n\033[1mKey Observations And Insights:\033[0;0m\n"
        "The pie chart reveals that 55.2% of respondents are locals, while the remaining 44.8% are tourists. This indicates that the majority of visitors to Universal Studios Singapore (USS) are locals.\n"
        "This might at first seem a little concerning as despite the small population size of Singapore (only around 6 million compared to billions in the world), it still accounts for the majority of visitor count to USS.\n"
        "We need to explore further as to why there might be insufficient tourists in USS.\n"
        "However, to determine whether this trend is a recent development or a long-standing pattern, it is essential to compare these proportions with historical data.\n"
        "Analyzing past visitor demographics will help assess any shifts in nationality distribution over time.\n")


  ### Analyse current visitor type by group of people respondents visit with ###
  print('\n\033[1m' + 'Analysing Number of Respondents by Visitor Type' + '\033[0;0m\n')
  visitor_type_count_current = get_visitor_type_count(df)
  plot_bar(visitor_type_count_current,
          title='Number Of Respondents Based On Visitor Type (Current Data)',
          xlabel='Visitor Type Categories',
          ylabel='Number of Respondents',
          color="violet")

  print("\n\033[1mKey Observations And Insights:\033[0;0m\n"
        "1) Largest group of visitors are families with young children.\n"
        "2) Guests who visit with friends form the next largest group, followed by familiy with teenagers.\n"
        "3) Solo travellers also form a sizeable group, making up 19% of the total survey respondants.\n"
        "Analyzing past visitor types will help us explore how the distribution of visitor types have shifted over time.\n")


  ### Analyse preferred time of day by current visitors ###
  print('\n\033[1m' + "Analysing Respondents' Preferred Time To Visit USS" + '\033[0;0m\n')
  print('\n\033[1m' + 'Category 1: When do visitors usually spend their day at USS?' + '\033[0;0m')
  occasion_count_current = get_occasion_count(df)
  plot_bar(occasion_count_current,
          title='Total Count of Preferred Day of Visitation (Current Data)',
          xlabel='Day Type',
          ylabel='Number Of Respondents',
          color='cornflowerblue')

  print("\n\033[1mKey Observations And Insights:\033[0;0m\n"
        "1) Weekdays have the largest number of respondants (138), indicating that the majority of visitors might choose a weekday to visit USS.\n"
        "   There might be several reasons such as avoiding large crowds on weekends and public holidays or that there might be certain promotions or discounts given on weekdays that entice customers to visit USS on a weekday instead.\n"
        "2) School Holidays also contribute to a significant number of responses (108) probably because there are many school children and teenagers who are only able to visit USS in this period due to their school/work days.\n"
        "3) Special Events account for the lowest number of respondants (77). This might be attributed to the fact that there are only some special events held in USS and some visitors might have visited other entertainment venues (not USS) in these occasions.\n")


  ### Analyse what time of day constitutes the highest demand ###
  print('\n\033[1m' + 'Category 2: What time of day constitutes the highest demand?' + '\033[0;0m\n')
  time_slot_count_current = get_time_slot_count(df)
  plot_bar(time_slot_count_current,
          title='Number Of Visitations Based On Time Periods (Current Data)',
          xlabel='Time Of Day',
          ylabel='Number Of Respondents',
          color='cornflowerblue')

  print("\n\033[1mKey Observations And Insights:\033[0;0m\n"
        "1) We observe that based on the length of the bars, Late Afternoon (2pm to 4pm) is the period of time where USS is the most crowded and has the most number of visitors.\n"
        "   This is followed by Evening (4pm to 6pm) and Early Afternoon (12pm to 2pm). We can conclude that the peak times of USS span from Midday to around 6pm in the evening.\n"
        "2) The least popular timing is Early Morning (8am to 10am) when USS just opens and might be still too early for some individuals who might be asleep at this timing.\n"
        "   Demand is not as high. Night (6pm to 9pm) also has a lower number of respondants as it might be too late for some individuals possibly due to the fact that they might have work/school the next day.\n")


  ### Analyse if special events are an important factor influencing visitors to visit USS ###
  print('\n\033[1m' + 'Category 3: Are special events an important factor influencing visitors to visit USS?' + '\033[0;0m\n')
  factor_count_current = get_factor_count(df)
  plot_bar(factor_count_current,
          title='Total Count of Each Factor Influencing Theme Park Visitation (Current Data)',
          xlabel='Factors',
          ylabel='Number Of Respondents',
          color=['magenta' if index == "Special Events" else 'cornflowerblue' for index in factor_count_current.index])

  print("\n\033[1mKey Observations And Insights:\033[0;0m\n"
        "The bar chart above suggests that `Special Events` may not be as important as other factors such as `Cost And Ticket Prices` and `Weather Conditions` in influencing survey respondants to visit theme parks.\n"
        "However, it is still a factor taken into consideration by 29% of respondants.\n")


  ### Analyse what special events attract visitors to USS ###
  print('\n\033[1m' + 'Category 4: What special events attract visitors to USS?' + '\033[0;0m\n')
  event_count_current = get_event_count(df)
  plot_bar(event_count_current,
          title='Total Count of Events Attracting Visitors (Current Data)',
          xlabel='Events',
          ylabel='Number Of Respondents',
          color='royalblue')

  print("\n\033[1mKey Observations And Insights:\033[0;0m\n"
        "From the bar chart above, we observe that \x1B[3mHalloween Horror Night\x1B[23m is most likely to attract guests to visit USS, followed by \x1B[3mMinion Land Grand Opening\x1B[23m then \x1B[3mA Universal Christmas\x1B[23m.\n"
        "However, the difference in total count between the 3 events is not large. A point to note is that \x1B[3mMinion Land Grand Opening\x1B[23m is a one-time event while \x1B[3mHalloween Horror Night\x1B[23m and \x1B[3mA Universal Christmas\x1B[23m are annual events.\n"
        "On the other hand, about 36% of survey respondents are not interested in any of the above events.\n")


  ### Analyse overall guest satisfaction ratings ###
  print('\n\033[1m' + 'Analysing Guest Satisfaction Rating' + '\033[0;0m\n')
  satisfaction_count_current = get_satisfaction_count(df)
  plot_bar(satisfaction_count_current,
          title='Overall Satisfaction Score Of USS (Current Data)',
          xlabel='Satisfaction Score (1-10)',
          ylabel='Number Of Respondents',
          color='plum',
          rotation=0)

  print("\n\033[1mKey Observations And Insights:\033[0;0m\n"
        "1) Referring to the Net Promoter Score (NPS) metric, 95 out of 507 survey respondents are promoters (gave a rating of 9 or 10), suggesting that only about 18.8% of them were satisfied with their experience at USS.\n"
        "2) About 53% of survey respondents gave scores of 7 and 8, which means that they would not strongly recommend others to visit USS.\n"
        "   This indicates that slightly over half of the survey respondents were not very satisfied with their experience at USS, leading to an indifferent score.\n")

"""++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

### Step 2: Obtain And Analyze The Demographics Of Visitors Who Visited USS Some Time Ago

After analyzing the demographics of current visitors to Universal Studios Singapore (USS), we will now extend our analysis to past visitors.

To achieve this, we will utilize a survey dataset from Kaggle, which contains visitor data for various Universal Studios locations worldwide from 2010 to 2021 (spanning approximately 4 to 14 years ago). The dataset is made by combining all the reviews on Trip Advisor for the various Universal Studios around the world. This dataset will serve as our historical reference for examining visitor demographics and satisfaction scores. However, before proceeding with the analysis, we must first clean and preprocess the data to ensure accuracy and reliability.

The dataset contains 12 columns:

1. `reviewer` - The name of the reviewer (might not be real name)
2. `Gender` - The gender of the reviewer
3. `Age Range` - The age range of the reviewer
4. `Tourist or Local` - Whether the reviewer is a tourist or a local
5. `branch` - The Universal Studios branch the reviewer went
6. `time_of_day` - The time the reviewer prefers to go to USS
7. `visitor_type` - Who does the visitor go with (Family, Friends, Alone etc.)
8. `day_preferred` - What day does the reviewer prefer to go to USS
9. `written-date` - The time when the reviewer wrote the review
10. `rating` - The rating the reviewer gave to his/her Universal Studios Experience
11. `title` - The title the reviewer wrote on Trip Advisor
12. `review_text` - The review description that the reviewer gave on Trip Advisor

Let us examine the dataset more:

We can first find the total number of rows of the dataset using the `shape` method.

To clean the raw dataset from Kaggle, we need to do the following:

*   Remove Unnecessary Columns - `reviewer`, `title`, `review_text` and `written_date`
*   Filter The Dataset For Rows Where `branch` is "Universal Studios Singapore" as we are not interested in other Universal Studios.
*   Rename Column Names For Consistency (`Gender` to `gender`, `Tourist or Local` to `nationality`, `Age Range` to `age_range`)
*   Rearrange The Columns Of The Dataset Where The Order Is - `gender`, `age_range`, `nationality`, `branch`, `visitor_type`, `day_preferred`, `time_of_day`, `rating`)
"""

# Clean historical reviews dataset
def clean_df_history(df_history):
  '''
  Args:
    df_history: DataFrame containing historical TripAdvisor reviews
  Returns:
    df_history: Cleaned DataFrame containing dataset
  '''
  df_history = df_history.copy()

  # Remove unnecessary columns
  df_history = df_history.drop(columns=['reviewer', 'title', 'review_text'])

  # Filter for rows where 'branch' is "Universal Studios Singapore"
  df_history = df_history[df_history['branch'] == "Universal Studios Singapore"]

  # Rename columns for consistency
  df_history = df_history.rename(columns={
      'Gender': 'gender',
      'Tourist or Local': 'nationality',
      'Age Range': 'age_range'
  })

  # Rearrange the columns in the specified order
  df_history = df_history[['gender', 'age_range', 'nationality', 'branch', 'visitor_type',
                          'day_preferred', 'time_of_day', 'rating', 'written_date']]

  # Get year and month of review
  df_history['year'] = df_history['written_date'].dt.year
  df_history['month'] = df_history['written_date'].dt.month

  # Drop written_date column
  df_history = df_history.drop(columns='written_date')

  # Reset the index and start from 1 instead of 0
  df_history = df_history.reset_index(drop=True)
  df_history.index += 1

  # Replace 'Families' with 'Family' in historical data for consistency
  df_history['visitor_type'] = df_history['visitor_type'].replace({
      'Families With Young Children': 'Family With Young Children',
      'Families With Teenagers': 'Family With Teenagers',
      'Families With Elderly': 'Family With Elderly'
  })

  # Replace 'time_of_day' column in historical data for consistency
  df_history['time_of_day'] = df_history['time_of_day'].replace({
    "Early Morning (8am To 10am)": "Early Morning (8am to 10am)",
    "Late Morning (10am To 12pm)": "Late Morning (10am to 12pm)",
    "Early Afternoon (12pm To 2pm)": "Early Afternoon (12pm to 2pm)",
    "Late Afternoon (2pm To 4pm)": "Late Afternoon (2pm to 4pm)",
    "Evening (4pm To 6pm)": "Evening (4pm to 6pm)",
    "Night (6pm To 9pm)": "Night (6pm to 9pm)"
})

  return df_history

"""The number of columns of the dataset is reduced to 10 - which tallies with the columns of `gender`, `age_range`, `nationality`, `branch`, `visitor_type`, `day_preferred`, `time_of_day`, `rating`, `year` and `month`. The number of rows has also been reduced significantly from 50904 rows to 15754 rows due to the fact that we only kept rows containing survey responses from Universal Studios Singapore. This also agrees with the fact that the original dataset contains only about 35% of reviews from Universal Studios Singapore.

Using this cleaned dataset, we can now proceed to analyze the demographics and satisfaction rates of customers who visited USS some time ago.
"""

# Plot line graph of number of reviews received each month from 2010 to 2021
def plot_line(df, title, xlabel, ylabel):
  '''
  Args:
    df: DataFrame containing dataset
    title: Title of the plot
    xlabel: Label for x-axis
    ylabel: Label for y-axis
  '''
  # Group by year and month and count occurrences
  monthly_counts = df.groupby(['year', 'month']).size().reset_index(name='count')

  # Pivot for easier plotting
  pivot_counts = monthly_counts.pivot(index='month', columns='year', values='count')

  # Plot line graph
  plt.figure(figsize=(12, 6))
  sns.lineplot(data=monthly_counts, x='month', y='count', hue='year',
              marker='o', palette='deep', linewidth=2)

  plt.title(title, fontsize=14)
  plt.xlabel(xlabel, fontsize=12)
  plt.ylabel(ylabel, fontsize=12)
  plt.xticks(range(1, 13),
            ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
  plt.grid(True, linestyle='--', alpha=0.6)
  plt.legend(title='Year', fontsize=10, bbox_to_anchor=(1.1, 1.02))
  plt.tight_layout()

  plt.show()

"""#### Step 2 Helper Function"""

# Step 2: Obtain And Analyze The Demographics Of Visitors Who Visited USS Some Time Ago
def analyse_reviews(df_history):
  '''
  Args:
    df_history: DataFrame of historical TripAdvisor Reviews
  '''

  print('\n\033[1m' + 'Step 2: Obtain And Analyze The Demographics Of Visitors Who Visited USS Some Time Ago' + '\033[0;0m\n')


  ### Analyse historical visitor demographic by age range ###
  print('\n\033[1m' + 'Analysing Number of Reviewers by Age Range' + '\033[0;0m\n')
  age_count_history = get_age_count(df_history, is_current=False)
  plot_bar(age_count_history,
          title='Number Of Reviewers Based On Age Range (Historical Data)',
          xlabel='Age Range',
          ylabel='Number of Reviewers')

  print("\n\033[1mKey Observations And Insights:\033[0;0m\n"
        "1) We observe that contrary to the current visitor dataset where 21 to 34 years old (Young Adults) forms the majority of the reviews, the historical visitor data reveals that the majority of visitors are in the age range of 13 to 20 years old (Teenagers).\n"
        "2) The age range of 21 to 34 years old still form a significant proportion of the visitors in the historical data. The downward trend from the category of 21 to 34 years old to 65 years old and above is similar to that of the current survey data.\n"
        "3) The age group that has the lowest number of respondants is still 65 years and above, similar to the current survey data.\n")


  ### Analyse historical visitor demographic by gender ###
  print('\n\033[1m' + 'Analysing Number of Reviewers by Gender' + '\033[0;0m\n')
  gender_count_history = get_gender_count(df_history, is_current=False)
  plot_pie(gender_count_history,
          title='Proportion Of Male And Female Visitors (Historical Data)',
          colors=['#377eb8', '#ff69b4'])

  print("\n\033[1mKey Observations And Insights:\033[0;0m\n"
        "Contrary to the current visitor survey responses where Females make up the majority of data, for the historical visitor reviews, Male (67.7%) makes up most of the data as compared to Female (32.3%).\n"
        "This shows that over time, there might either be:\n"
        "1) More Females interested in visiting USS\n"
        "2) Less Males interested in visiting USS\n"
        "3) Both (1) and (2)\n"
        "\nWe will discuss more about this in Steps 3 and 4 of the question.\n")


  ### Analyse historical visitor demographic by nationality ###
  print('\n\033[1m' + 'Analysing Number of Reviewers by Nationality' + '\033[0;0m\n')
  nationality_count_history = get_nationality_count(df_history, is_current=False)
  plot_pie(nationality_count_history,
          title='Proportion Of Locals And Tourists (Historical Data)',
          colors=['#AEC6CF', '#FA8072'])

  print("\n\033[1mKey Observations And Insights:\033[0;0m\n"
        "Contrary to the current visitor survey responses where Locals make up the majority of data, for the historical visitor reviews, Tourists (65.7%) makes up most of the data as compared to Locals (34.3%).\n"
        "This shows that over time, there might either be:\n"
        "1) More Locals interested in visiting USS\n"
        "2) Less Tourists interested in visiting USS\n"
        "3) Both (1) and (2)\n"
        "\nWe will discuss more about this in Steps 3 and 4 of the question.\n")


  ### Analyse historical visitor type by group of people respondents visit with ###
  print('\n\033[1m' + 'Analysing Number of Reviewers by Visitor Type' + '\033[0;0m\n')
  visitor_type_count_history = get_visitor_type_count(df_history, is_current=False)
  plot_bar(visitor_type_count_history,
          title='Number Of Reviewers Based On Visitor Type (Historical Data)',
          xlabel='Visitor Categories',
          ylabel='Number of Reviewers',
          color='violet')

  print("\n\033[1mKey Observations And Insights:\033[0;0m\n"
        "1) We observe that there is a significantly higher proportion of visitors who are `Families With Young Children`, as compared to the survey responses where the difference in proportion was not as significant.\n"
        "2) There is a higher proportion of `Families With Elderly` as compared to `Solo Travellers` and `Families With Teenagers`, which differs from our earlier analysis based on recent survey responses.\n"
        "These differences could be explained by possible reasons below:\n"
        " 1) Distributions of visitor types have shifted over the years due to marketing strategies or change in preferences\n"
        " 2) Limited survey outreach which resulted in skewed survey responses\n"
        " 3) Both (1) and (2)\n"
        "\nWe will discuss more about this in Steps 3 and 4 of the question.\n")


  ### Analyse preferred time of day by past visitors ###
  print('\n\033[1m' + "Analysing Reviewers' Preferred Time To Visit USS" + '\033[0;0m\n')
  print('\n\033[1m' + 'Category 1: When do visitors usually spend their day at USS?' + '\033[0;0m')
  occasion_count_history = get_occasion_count(df_history, is_current=False)
  plot_bar(occasion_count_history,
          title='Total Count of Preferred Day of Visitation (Historical Data)',
          xlabel='Day Categories',
          ylabel='Number Of Reviewers',
          color='cornflowerblue')

  print("\n\033[1mKey Observations And Insights:\033[0;0m\n"
        "1) For the historical data, the majority of respondants chose Weekends as their preferred days, in contrast to Weekdays for the current data. There might be a shift in the visitor' preferences from Weekends to Weekdays.\n"
        "2) The number of reviewers who prefer visiting USS during Special Events is the least among the other options, similar to the current dataset. However, contrary to the current dataset, the historical data shows much fewer visitors whose preference is on Special Events.\n"
        "3) The proportion of visitors who chose School Holidays as their preferred days to visit USS for the historic dataset is significantly lower compared to the current dataset.\n")

  plot_line(df_history,
            title='Number of Reviews Each Month (Historical Data)',
            xlabel='Month',
            ylabel='Number of Reviews')

  print("\n\033[1mKey Observations And Insights:\033[0;0m\n"
        "The graph shows that there tends to be spike in the number of reviews in the months of July and December to January for most years, with the exception of 2020 and 2021 when the pandemic was still rampant.\n"
        "This coincides with the long school holiday period in Singapore (June and December), suggesting that a larger number of people tend to visit USS during the holidays, even if it may not be their preferred choice.\n")


  ### Analyse what time of day constitutes the highest demand ###
  print('\n\033[1m' + 'Category 2: What time of day constitutes the highest demand?' + '\033[0;0m\n')
  time_slot_count_history = get_time_slot_count(df_history, is_current=False)
  plot_bar(time_slot_count_history,
          title='Number of Reviewers Based on Preferred Timings (Historical Data)',
          xlabel='Time of Day',
          ylabel='Number Of Reviewers',
          color='cornflowerblue')

  print("\n\033[1mKey Observations And Insights:\033[0;0m\n"
        "1) For the historical data, the majority of respondants chose Evening (4pm to 6pm) as their preferred timings to visit USS, in contrast to Late Afternoon (2pm to 4pm) for the current data.\n"
        "   However, Late Afternoon (2pm to 4pm) still forms quite a significant proportion of visitors favorite timing for the historical data.\n"
        "2) The number of reviewers who prefer visiting USS during Early Morning (8am to 10am) and Night (6pm to 9pm) is the least among the other options, similar to the current dataset.\n"
        "3) The number of respondants still shows an increase from Early Morning (8am to 10am) to Late Afternoon (2pm to 4pm), similar to the current dataset.\n")


  ### Analyse if special events are an important factor influencing visitors to visit USS ###
  print('\n\033[1m' + 'Analysing If Special Events Attracted Past Visitors' + '\033[0;0m\n')
  occasion_count_history = get_occasion_count(df_history, is_current=False)
  plot_bar(occasion_count_history,
          title='Number of Reviewers Based on Preferred Days (Historical Data)',
          xlabel='Day Categories',
          ylabel='Number Of Reviewers',
          color=['magenta' if index == "Special Events" else 'royalblue' for index in occasion_count_history.index])

  print("\n\033[1mKey Observations And Insights:\033[0;0m\n"
        "Similar to current survey visitor responses, the number of reviewers who prefer visiting USS during Special Events is the least among the other options.\n"
        "However, we observe that this difference between Special Events and other options is more significant in the historical data as compared to data obtained from recent survey responses.\n"
        "This might be due to the fact that there might be more advertising of these special events by USS in recent years that led to greater demand for USS during these events.\n")


  ### Analyse past overall guest satisfaction ratings ###
  """
  NOTE: The rating column is measured from 1 to 5, which is different from the scale from 1 to 10 in the satisfaction score for the current dataset.
        We will assume that a rating of 4 in the historic dataset will translate to a rating of 8 in the current dataset.
  """
  print('\n\033[1m' + 'Analysing Historical Guest Satisfaction Rating' + '\033[0;0m\n')
  satisfaction_count_history = get_satisfaction_count(df_history, is_current=False)
  plot_bar(satisfaction_count_history,
          title='Distribution of Ratings (Historical Data)',
          xlabel='Ratings (1-5)',
          ylabel='Number Of Reviewers',
          color='plum',
          rotation=0)

  print("\n\033[1mKey Observations And Insights:\033[0;0m\n"
        "1) As seen from the bar chart above, over 83% of reviewers gave USS a rating of 4 and 5 on TripAdvisor, suggesting that they were generally satisfied with their experience.\n"
        "2) A minority (5.4%) expressed their dissatisfaction by giving ratings of 1 and 2.\n"
        "3) The historical ratings of USS are generally higher compared to the satisfaction ratings of USS currently.\n"
        "   (the mode rating for historic data is 5 but the mode rating for the current USS is 7/10, which is only between 3 and 4 when translated to a scale of 5)\n"
        "   This means that overall satisfaction of USS fell in the recent years.\n")

"""++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

### Step 3: Compare The Visitor Demographics + Identify Trends And Majority Classes

For effective comparisons of visitor demographics, we can make use of heatmaps for every demographic section. We will be comparing proportions in the form of percentages (in 1 decimal place for all heatmapsfor consistency) instead of counts as the number of survey responses in the historic data far exceeds the number of survey responses in the current data (almost 15000 vs 500).

The sections that we will be comparing are:

**1.   Age Of Visitors**

**2.   Gender Of Visitors**

**3.   Nationality Of Visitors**

**4.   Visitor Type**

**5.   Preferred Days**

**6.   Preferred Timing**
"""

# Plot heatmap to compare current and historical data
def comparison_heatmap(current_data, historical_data, title, cmap="Blues"):
  '''
  Args:
    current_data: Series containing current data counts
    historical_data: Series containing historical data counts
    title: Title of the heatmap
    cmap: Colormap to use for the heatmap
  '''

  # Compute proportions
  current_proportions = (current_data / current_data.sum() * 100).round(1)
  historic_proportions = (historical_data / historical_data.sum() * 100).round(1)

  # Combine into a DataFrame
  heatmap_data = pd.DataFrame({'Current Data': current_proportions, 'Historical Data': historic_proportions})
  annot_data = heatmap_data.astype(str) + "%"

  # Create heatmap
  plt.figure(figsize=(8, 6))
  sns.heatmap(heatmap_data, annot=annot_data, cmap=cmap, fmt="", linewidths=0.5)
  plt.title(title)
  plt.tight_layout()
  plt.show()

"""#### Step 3 Helper Function"""

# Step 3: Compare the Visitor Demographics + Identify Trends and Majority Classes
def compare(df, df_history):
  '''
  Args:
    df: DataFrame of survey responses
    df_history: DataFrame of historical TripAdvisor Reviews
  '''


  print('\n\033[1m' + 'Step 3: Compare the Visitor Demographics & Identify Trends and Majority Classes' + '\033[0;0m\n')


  ### Analyse change in proportions of visitors' ages ###
  print('\n\033[1m' + "Analyse Changes in Proportions of Visitors' Ages" + '\033[0;0m\n')
  age_count_current = get_age_count(df)
  age_count_history = get_age_count(df_history, is_current=False)
  comparison_heatmap(age_count_current,
                   age_count_history,
                   title="Age Group Proportions: Current vs. Historical Data",
                   cmap="Reds")

  print("This heatmap shows the age distribution of visitors comparing current versus historical data for USS.\n"
        "\nChildren (Below 12 Years Old):\n"
        "- Current: 16.8%\n"
        "- Historical: 15.3%\n"
        "- Slight increase in children visiting currently\n"
        "\nTeenagers (13 To 20 Years Old):\n"
        "- Current: 25.0%\n"
        "- Historical: 24.7%\n"
        "- Almost unchanged, representing about a quarter of visitors in both periods\n"
        "\nYoung Adults (21 To 34 Years Old):\n"
        "- Current: 32.3%\n"
        "- Historical: 20.3%\n"
        "- Significant increase of 12 percentage points, making this the most dramatic change in the dataset\n"
        "\nMiddle-Aged Adults (35 To 49 Years Old):\n"
        "- Current: 14.3%\n"
        "- Historical: 19.6%\n"
        "- Notable decrease of about 5.3 percentage points\n"
        "\nOlder Adults (50 To 64 Years Old):\n"
        "- Current: 7.7%\n"
        "- Historical: 15.3%\n"
        "- Substantial decrease of 7.6 percentage points, showing this demographic has become much less represented\n"
        "\nSeniors (65 Years Old And Above):\n"
        "- Current: 4.0%\n"
        "- Historical: 4.8%\n"
        "- Slight decrease, but remains the smallest proportion in both datasets\n"

        "\n\033[1mKey Trends:\033[0;0m\n"
        "- The visitor population has significantly shifted toward younger adults (21-34), which now make up nearly a third of all visitors.\n"
        "- There has been a marked decline in middle-aged and older adult visitors (35-64 age groups).\n"
        "- The proportion of children and teenagers remains relatively stable between the two periods.\n"
        "- The current visitor profile is more heavily skewed toward younger demographics, with 74.1% of visitors under 35 years old (compared to 60.3% historically).\n"
        "- The 50+ age groups collectively dropped from 20.1% historically to just 11.7% currently.\n"
        "\nThis suggests a strategic repositioning or changing appeal of USS, with significantly stronger attraction to young adults and slightly reduced engagement with older demographics.\n")


  ### Analyse change in proportions of visitors' gender ###
  print('\n\033[1m' + "Analyse Changes in Proportions of Visitors' Gender" + '\033[0;0m\n')
  gender_count_current = get_gender_count(df)
  gender_count_history = get_gender_count(df_history, is_current=False)
  comparison_heatmap(gender_count_current,
                   gender_count_history,
                   title="Gender Proportions: Current vs. Historical Data",
                   cmap="Greens")

  print("This heatmap displays gender proportions comparing current versus historical data for USS.\n"
        "\nThe gender distribution shows a dramatic reversal between the two time periods:\n"
        "\nCurrent Data:\n"
        "- Female: 61.0%\n"
        "- Male: 39.0%\n"
        "\nHistorical Data:\n"
        "- Female: 32.3%\n"
        "- Male: 67.7%\n"

        "\n\033[1mKey Trends:\033[0;0m\n"
        "- There has been a complete inversion in the gender proportions between the historical and current periods.\n"
        "  Historically, males dominated the visitor demographics, representing over two-thirds (67.7%) of all visitors.\n"
        "- Currently, females constitute the clear majority at 61.0% of visitors.\n"
        "  The shift represents a 28.7 percentage point increase in female representation.\n"
        "- Male representation has correspondingly decreased by 28.7 percentage points.\n"

        "\nThis significant gender proportion reversal suggests a possible fundamental change in either:\n"
        "The attraction's appeal or marketing strategy, the programming or exhibition content, targeted outreach efforts or the organizational culture or reputation.\n")


  ### Analyse change in proportion of visitors' nationalities ###
  print('\n\033[1m' + "Analyse Changes in Proportions of Visitors' Nationalities" + '\033[0;0m\n')
  nationality_count_current = get_nationality_count(df)
  nationality_count_history = get_nationality_count(df_history, is_current=False)
  comparison_heatmap(nationality_count_current,
                   nationality_count_history,
                   title="Nationality Proportions: Current vs. Historical Data")

  print("This heatmap shows the nationality proportions comparing current versus historical data for USS, breaking down visitors into Local and Tourist categories.\n"
        "\nCurrent Data:\n"
        "- Local: 55.2%\n"
        "- Tourist: 44.8%\n"
        "\nHistorical Data:\n"
        "- Local: 34.3%\n"
        "- Tourist: 65.7%\n"

        "\n\033[1mKey Trends:\033[0;0m\n"
        "- There has been a significant shift in the visitor composition, with the proportions essentially inverting between the two time periods.\n"
        "- Historically, tourists were the dominant visitor group, making up nearly two-thirds (65.7%) of all visitors.\n"
        "  Currently, locals constitute the majority at 55.2% of visitors.\n"
        "- This represents a 20.9 percentage point increase in local visitation.\n"
        "  Tourist representation has correspondingly decreased by 20.9 percentage points.\n"

        "\nThis substantial change in the visitor demographic suggests:\n"
        "- A shift in focus toward attracting and serving the local community\n"
        "- Potentially reduced international or out-of-area tourism or a possible reorientation of programming or exhibits to appeal more to local interests.\n"
        "- There could also be changes in marketing strategy to target the local market.\n")


  ### Analyse changes in proportions of visitor types ###
  print('\n\033[1m' + "Analyse Changes in Proportions of Visitor Types" + '\033[0;0m\n')
  visitor_type_count_current = get_visitor_type_count(df)
  visitor_type_count_history = get_visitor_type_count(df_history, is_current=False)
  comparison_heatmap(visitor_type_count_current,
                   visitor_type_count_history,
                   title="Visitor Type Proportions: Current vs. Historical Data",
                   cmap="Purples")

  print("This heatmap illustrates the visitor type proportions comparing current versus historical data for USS, breaking down visitors into five distinct categories.\n"
        "\nCurrent Data:\n"
        "- Solo Traveller: 19.1%\n"
        "- Visiting With Friends: 25.8%\n"
        "- Family With Young Children: 29.0%\n"
        "- Family With Teenagers: 20.7%\n"
        "- Family With Elderly: 5.3%\n"
        "\nHistorical Data:\n"
        "- Solo Traveller: 13.0%\n"
        "- Visiting With Friends: 19.0%\n"
        "- Family With Young Children: 37.2%\n"
        "- Family With Teenagers: 12.9%\n"
        "- Family With Elderly: 18.0%\n"

        "\n\033[1mKey Trends:\033[0;0m\n"
        "1) Family With Young Children remains the largest visitor segment in both periods, though it has decreased by 8.1 percentage points (from 37.2% to 29.1%).\n"
        "2) Visiting With Friends has increased significantly from 19.0% to 25.9%, suggesting USS has become more popular as a social destination.\n"
        "3) Family With Elderly has seen the most dramatic decrease, dropping from 18.0% to just 5.3% (a 12.7 percentage point decline).\n"
        "4) Family With Teenagers has increased substantially from 12.9% to 20.8%, an 8.1 percentage point increase.\n"
        "5) Solo Traveller visitation has increased from 13.0% to 18.8%.\n"

        "\nThese changes align with the previous demographic shifts observed:\n"
        "- The decline in elderly family visits corresponds with the overall reduction in older visitors.\n"
        "- The increase in friend groups and solo travelers aligns with the shift toward younger adults (21-34).\n"
        "- The growth in teenage family visits matches the stable proportion of teenage visitors we saw in the age breakdown.\n"
        "\nOverall, USS has transitioned from being primarily a family destination (particularly for young children and elderly family members) to a more diversified attraction that appeals to varied groups, especially friends, solo visitors and families with teenagers.\n")


  ### Analyse changes in proportions of preferred days of visit ###
  print('\n\033[1m' + "Analyse Changes in Proportions of Visitors' Preferred Days of Visit" + '\033[0;0m\n')
  occasion_count_current = get_occasion_count(df)
  occasion_count_history = get_occasion_count(df_history, is_current=False)
  comparison_heatmap(occasion_count_current,
                   occasion_count_history,
                   title="Preferred Day Proportions: Current vs. Historical Data",
                   cmap="Oranges")

  print("This heatmap displays the preferred day proportions comparing current versus historical data for USS, showing when visitors prefer to attend.\n"
        "\nCurrent Data:\n"
        "- Weekdays: 27.4%\n"
        "- Weekends: 19.5%\n"
        "- Public Holidays: 16.3%\n"
        "- School Holidays: 21.5%\n"
        "- Special Events: 15.3%\n"
        "\nHistorical Data:\n"
        "- Weekdays: 24.0%\n"
        "- Weekends: 29.6%\n"
        "- Public Holidays: 28.1%\n"
        "- School Holidays: 14.9%\n"
        "- Special Events: 3.4%\n"

        "\n\033[1mKey Trends:\033[0;0m\n"
        "1) Visitor timing preferences have significantly shifted between the two periods, with much more evenly distributed attendance patterns now.\n"
        "2) Historically, attendance was concentrated during weekends (29.6%) and public holidays (28.1%), which together accounted for 57.7% of visitors.\n"
        "3) Currently, weekdays have become the most popular time to visit (27.4%), followed by school holidays (21.5%).\n"
        "4) Weekend visitation has decreased substantially from 29.6% to 19.5% (a 10.1 percentage point decline).\n"
        "5) Public holiday visitation has dropped significantly from 28.1% to 16.3% (an 11.8 percentage point decrease).\n"
        "6) Special events have seen a dramatic increase in popularity, rising from just 3.4% to 15.3% (an 11.9 percentage point increase).\n"
        "7) School holiday visitation has increased from 14.9% to 21.5% (a 6.6 percentage point rise).\n"

        "\nThese changes align with the demographic shifts observed in previous charts:\n"
        "- The increase in weekday and special event visitation correlates with the rise in local visitors who likely have more flexibility to visit on non-peak days.\n"
        "- The growth in school holiday attendance matches the increase in families with teenagers.\n"
        "- The decrease in weekend and public holiday visitation suggests less reliance on tourism and out-of-town visitors.\n"

        "\nOverall, USS has transitioned from a destination primarily visited during traditional peak tourism periods (weekends and public holidays) to one with more balanced attendance throughout different time periods, with special emphasis on weekdays, school holidays, and special events.\n")


  ### Analyse changes in proportions of preferred time of visit ###
  print('\n\033[1m' + "Analyse Changes in Proportions of Visitors' Preferred Time of Visit" + '\033[0;0m\n')
  time_slot_count_current = get_time_slot_count(df)
  time_slot_count_history = get_time_slot_count(df_history, is_current=False)
  comparison_heatmap(time_slot_count_current,
                    time_slot_count_history,
                    title="Preferred Time Slot Proportions: Current vs. Historical Data",
                    cmap="Greys")

  print("This heatmap presents the preferred time slot proportions comparing current versus historical data for USS, showing when during the day visitors prefer to attend.\n"
        "\nCurrent Data:\n"
        "- Early Morning (8am to 10am): 13.8%\n"
        "- Late Morning (10am to 12pm): 16.4%\n"
        "- Early Afternoon (12pm to 2pm): 17.7%\n"
        "- Late Afternoon (2pm to 4pm): 20.0%\n"
        "- Evening (4pm to 6pm): 17.9%\n"
        "- Night (6pm to 9pm): 14.3%\n"
        "\nHistorical Data:\n"
        "- Early Morning (8am to 10am): 14.6%\n"
        "- Late Morning (10am to 12pm): 15.0%\n"
        "- Early Afternoon (12pm to 2pm): 16.1%\n"
        "- Late Afternoon (2pm to 4pm): 18.4%\n"
        "- Evening (4pm to 6pm): 21.4%\n"
        "- Night (6pm to 9pm): 14.6%)\n"

        "\n\033[1mKey Trends:\033[0;0m\n"
        "1) The time slot preferences show relatively modest changes compared to the more dramatic shifts observed in other demographic categories.\n"
        "2) Late afternoon (2pm to 4pm) has become the most popular time slot currently at 20.0%, while evening (4pm to 6pm) was historically the most popular at 21.4%.\n"
        "3) Evening visitation has decreased from 21.4% to 17.9% (a 3.5 percentage point decline).\n"
        "4) The current distribution is more balanced across time slots, with no single time period strongly dominating.\n"
        "5) Early afternoon and late morning have seen modest increases in popularity (1.6 and 1.4 percentage points respectively).\n"
        "6) Early morning and night time slots remain the least popular periods in both datasets, with minimal changes.\n"

        "\nThese time slot preference changes align with the broader demographic shifts:\n"
        "- The more evenly distributed time preferences may reflect the greater proportion of local visitors who have more flexibility in their visiting times.\n"
        "- The slight decline in evening visits could relate to the decrease in tourist visitors, who might previously have preferred to visit attractions later in the day.\n"
        "- The increased popularity of afternoon slots aligns with the growth in family and friend group visits.\n")

"""### Step 4: Suggest Effective Strategies Tailored To Identified Groups

#### Step 4 Helper Function
"""

# Step 4: Suggest Effective Strategies Tailored To Identified Groups
def suggestions():


  print('\n\033[1m' + 'Step 4: Suggest Effective Strategies Tailored To Identified Groups' + '\033[0;0m\n')


  ### Introduction ###
  print("Before we propose effective strategies, we need to first determine the groups that are under-represented in the survey.\n"
        "This is to entice the groups of individuals who might initally not be interested in visiting USS in order to increase the overall demand for USS, as well as enhancing the guest experience at USS.\n"

        "\nFrom the results obtained above, we can pinpoint certain groups that might not be visiting USS as much.\n"
        "1) The Elderly (Particularly 65 years and above)\n"
        "2) Males (Dropping in proportions recently)\n"
        "3) Tourists (Dropping in proportions recently)\n"

        "\nThere are other groups of people that although still have fairly high proportions, it shows a downward trend from previous years.\n"
        "USS should also propose marketing strategies to entice these groups of people to continue visiting USS. These groups include:\n"
        "1) Families With Young Children (Dropped in proportions though still high)\n"
        "2) 35 To 49 Year Olds (Working Adults)\n")


  ### Propose strategies targeting the elderly ###
  print('\n\033[1m' + 'Proposing Strategies Targeting the Elderly' + '\033[0;0m\n')

  print("1. Senior Discount Days: Offer exclusive senior citizen discounts on weekdays or off-peak periods (e.g., Senior Citizens’ Special: 20% Off Tickets Every Wednesday)\n"
        "- Offering senior discounts during off-peak times can help fill up the park during slower periods, increasing overall attendance.\n"
        "- By targeting seniors, USS attracts a demographic that might otherwise avoid peak pricing. Seniors visiting on discounted days are likely to spend on additional services such as food, beverages, merchandise, and special experiences.\n"
        "  The sense of savings from the discount encourages seniors to purchase tickets and additional items that they might not have otherwise.\n"

        "\n2. Group Discounts for Senior Groups: Provide group pricing for seniors visiting with family or friends\n"
        "- Offering group discounts encourages seniors to visit USS with their families or friends, which can increase group ticket sales.\n"
        "- Group visits often lead to higher spending since more people are on-site, and each individual tends to purchase food, drinks, and souvenirs.\n"
        "- Additionally, seniors visiting with companions will likely make use of other park amenities (such as VIP tours, photo packages, and merchandise).\n"
        "- By offering a lower price for senior groups, USS can attract larger parties, thus improving both ticket sales and overall park revenue through group purchases and shared experiences.\n"

        "\n3. Relaxation Zones: Designate calming rest areas with comfortable seating, shade, and gentle music for seniors to relax in between attractions\n"
        "- Relaxation zones enhance the experience for elderly visitors, encouraging them to stay longer in the park without feeling fatigued.\n"
        "  This added comfort can lead to increased spending on food, drinks, and souvenirs as seniors take breaks in these areas.\n"
        "- The zones also provide a sense of exclusivity and care, which could result in more seniors choosing USS as their destination for leisure, leading to repeat visits.\n"
        "- Creating a positive and comfortable environment for older guests can also foster word-of-mouth promotion, attracting new visitors from within the same demographic, boosting future revenue.\n"

        "\n4. Senior-Friendly Tours: Offer guided, leisurely-paced tours around the park\n"
        "- Senior-friendly tours can be monetized by charging an additional fee for guided experiences, adding another revenue stream to USS.\n"
        "- These tours cater specifically to the interests and pace of elderly visitors, which not only enhances their experience but also offers them something they might be willing to pay for.\n"
        "- Seniors often appreciate leisurely, informative tours that are easy to follow, and offering them the chance to learn about the history of USS.\n"

        "\n5. Accessible Services: Promote accessibility options like wheelchair rentals, mobility aids, and priority seating for shows\n"
        "- Providing accessible services like wheelchair rentals and priority seating ensures that elderly visitors, especially those with mobility issues, can fully enjoy the park without feeling restricted.\n"
        "- The availability of mobility aids makes USS more inclusive, attracting a wider range of seniors, including those who may have previously thought the park was not accessible to them.\n"
        "- By offering these services, USS not only increases the likelihood of senior visits but also generates additional revenue from rentals and premium services like priority seating.\n")


  ### Propose strategies targeting males ###
  print('\n\033[1m' + 'Proposing Strategies Targeting Males' + '\033[0;0m\n')

  print("1. Increase Action-Packed Themed Attractions and Events: Focus on high-adrenaline, action-oriented rides and experiences, such as those based on blockbuster action movies (e.g., Fast & Furious, Transformers, Jurassic Park)\n"
        "- Men often enjoy intense, thrilling experiences. By emphasizing action-packed attractions, USS can increase ticket sales as these types of experiences tend to attract larger crowds.\n"
        "- Additionally, special events can be ticketed separately, generating extra revenue.\n"
        "- Merchandise related to these events (e.g., superhero or movie-themed items) can also drive increased sales.\n"
        "- High-adrenaline attractions also encourage repeat visits, especially if new features or events are added over time.\n"

        "\n2. Partnerships with Popular Male-Oriented Brands: Partner with well-known male-oriented brands (e.g., sports brands, gaming companies, or popular men's lifestyle brands)\n"
        "- Brand partnerships can draw new visitors who are loyal to specific brands, leading to increased foot traffic and engagement.\n"
        "- Co-branded merchandise (e.g., limited-edition items from a popular gaming brand) can be sold, providing additional revenue.\n"
        "- Additionally, exclusive partnerships often generate buzz and marketing opportunities, leading to greater awareness and attendance from the target demographic.\n"

        "\n3. Sports Bar and Gaming Lounges: Set up sports bars or gaming lounges within the park where male visitors can watch live sports events, play games, or relax while enjoying food and drinks\n"
        "- Sports bars and gaming lounges create an opportunity to increase F&B revenue by offering food and drink packages.\n"
        "- These venues are attractive for groups of male visitors who can socialize, relax, and spend time at the park in between attractions.\n"
        "- By creating a space for leisure, USS can encourage visitors to stay longer in the park, increasing their overall spending on food, drinks, and potentially event tickets or VIP access to exclusive viewing areas for major events.\n"

        "\n4. Adventure and Extreme Sports Experiences: Introduce extreme sports or adventure activities, such as virtual reality (VR) experiences, rock climbing, or even a high-flying experience like bungee jumping\n"
        "- Adventure and extreme sports experiences are often popular among men who seek thrilling, physically demanding activities.\n"
        "- Charging for these activities separately or offering package deals can drive additional revenue.\n"
        "  Such experiences can also increase the time visitors spend in the park, resulting in more spending on food, drinks, and merchandise.\n"
        "- These types of attractions also encourage repeat visits from thrill-seekers.\n")


  ### Propose strategies targeting tourists ###
  print('\n\033[1m' + 'Proposing Strategies Targeting Tourists' + '\033[0;0m\n')
  print("1. Launch International Marketing Campaigns: Partner with international travel agencies, airlines, and influencers to launch targeted marketing campaigns in key tourist markets such as China, India and Southeast Asia\n"
        "- International tourists typically spend more on experiences, food, and souvenirs when visiting a theme park.\n"
        "- By attracting more tourists from overseas, USS can tap into new revenue streams from foreign visitors who may also contribute to increased hotel bookings, travel packages, and extended stays.\n"
        "- This kind of exposure boosts ticket sales, as well as demand for premium experiences like VIP tours and fast passes, which increase overall spending per guest.\n"

        "\n2. Create Seasonal or Themed Events to Attract Tourists During Off-Peak Periods\n"
        "- Host special seasonal or limited-time themed events such as 'Summer Fest' or 'Halloween Horror Nights' to entice tourists during off-peak months.\n"
        "- Promoting these events through global tourism channels can attract visitors specifically interested in these unique experiences.\n"
        "- Special events generate excitement and urgency, encouraging tourists to visit USS during times when they might otherwise not have considered it.\n"
        "- These events can drive up ticket prices for exclusive access, increase the sale of themed merchandise, and boost spending on food and beverage items.\n"
        "  The demand for limited-time events can also increase park attendance during historically slower seasons, reducing revenue loss from lower off-peak visitation.\n"

        "\n3. Offer Multi-Day Passes with Exclusive Benefits for Tourists: Introduce multi-day passes specifically targeted at tourists, which allow for unlimited access to USS and partner attractions over a certain number of days\n"
        "- Multi-day passes encourage tourists to spend more time at USS, potentially increasing per-visitor revenue.\n"
        "- Tourists who purchase multi-day passes are more likely to spend on food, merchandise, and add-on services like express passes or special events.\n"
        "- By offering exclusive benefits, USS also incentivizes higher-priced purchases and repeat visits during the tourist's stay in Singapore.\n"

        "\n4. Develop Unique Tourist-Centric Experiences and Packages: Create unique experiences tailored to tourists - behind-the-scenes tours, private character meet-and-greets or exclusive access to new attractions\n"
        "- Unique, premium experiences can command higher prices and attract tourists willing to pay extra for exclusive access.\n"
        "- Offering tailored experiences encourages tourists to spend more than they would on standard tickets, boosting overall revenue per guest.\n"
        "- These high-value experiences can also be marketed as limited-time offers, increasing demand and encouraging tourists to visit USS sooner rather than later.\n"
        "- Additionally, the luxury appeal of these experiences may attract high-net-worth tourists, further increasing revenue from premium offerings.\n")


  ### Propose strategies targeting families with young children ###
  print('\n\033[1m' + 'Proposing Strategies Targeting Families With Young Children' + '\033[0;0m\n')
  print("1. Family Passes: Offer bundles that include tickets for two adults and two children at a discounted rate.\n"
        "- Family bundles make the experience more affordable for families, which can lead to a higher volume of ticket sales.\n"
        "- Families are more likely to purchase bundles, as they feel they are getting better value for money.\n"
        "- The discounted rate encourages families to buy tickets for both parents and children at once, and they might also purchase additional services like food, merchandise, or express passes, boosting overall revenue.\n"
        "- Offering bundled pricing creates a sense of cost savings, increasing the likelihood of families committing to a visit.\n"

        "\n2. Free Entry for Toddlers: Provide free admission for children under 3 years old, making it more affordable for families.\n"
        "- While offering free admission to toddlers may seem like a loss of revenue for the very young group, it encourages families to visit USS in the first place.\n"
        "- With younger children not being charged for tickets, families with toddlers are more likely to attend.\n"
        "- These families will still contribute to revenue by spending on food, merchandise, photo ops, and additional experiences that are not discounted, which helps to offset the 'free' entry for the toddlers.\n"
        "- This approach also attracts more repeat visits as parents may return when they have multiple children of different age groups.\n"

        "\n3. Interactive Learning Stations: Set up engaging, hands-on experiences related to the movies and themes (eg movie-making workshops, or behind-the-scenes shows)\n"
        "     Seasonal Family Events: Host special themed events (eg Christmas or Halloween) with kid-friendly activities\n"
        "- Interactive and educational experiences are highly attractive to families, especially when they are tied to popular movie themes that kids love.\n"
        "- These types of experiences can be monetized through special admission fees or bundled with ticket sales as part of a premium package.\n"
        "- Additionally, interactive stations encourage parents and children to spend more time at the venue, which leads to increased spending on food, drinks, souvenirs, and other experiences.\n"
        "- Children's participation in workshops also leads to a memorable experience, increasing the likelihood of word-of-mouth recommendations and repeat visits, which drives future revenue.\n"

        "\n4. Character Meet-and-Greets: Feature more kid-friendly characters apart from the usual ones that are currently available in USS for family photos and interactions\n"
        "- Character meet-and-greets are a major draw for families with young children.\n"
        "- Offering popular, kid-friendly characters allows USS to charge a premium for exclusive photo opportunities or VIP meet-and-greet experiences.\n"
        "- Families are willing to pay for these experiences to create lasting memories and unique interactions for their children.\n"
        "- Additionally, character-themed merchandise can be sold at meet-and-greet locations, further increasing revenue from each family.\n"
        "- The emotional connection created by meeting beloved characters encourages families to visit more often, especially if they know new characters or themed experiences are coming up.\n")


  ### Propose strategies targeting 35 to 49 year olds (working adults) ###
  print('\n\033[1m' + 'Proposing Strategies Targeting 35 to 49 Year Olds (Working Adults)' + '\033[0;0m\n')
  print("1. After-Work or Late-Night Events: Host special after-work or late-night events that cater to working adults, such as 'Happy Hour' events, themed parties, or exclusive access to certain attractions or shows during the evening\n"
        "- These events can drive attendance during typically off-peak hours, filling the park when it may otherwise be less crowded.\n"
        "- By offering special events, USS can attract working adults who may not typically have the time to visit during the day, increasing overall ticket sales.\n"
        "- The addition of exclusive experiences like VIP access, premium shows, or food and beverage deals can encourage working adults to spend more, boosting revenue from premium packages, food, drinks, and merchandise.\n"
        "- The events can also include paid add-ons like drink tickets or special photo opportunities, further driving spending.\n"

        "\n2. Corporate Partnership Packages: Partner with businesses and corporations to offer exclusive corporate discounts or team-building packages for employees\n"
        "- Corporate partnerships can lead to bulk ticket sales, generating significant revenue from organizations buying tickets for groups of employees.\n"
        "- Team-building events or group outings create opportunities for USS to offer additional upsells, like lunch or dinner packages, private tours, or special events tailored to corporate groups.\n"
        "- The exposure to corporate clients may encourage them to visit USS for personal trips, bringing in new customers and increasing repeat visits.\n"
        "- Offering exclusive perks or experiences for employees can elevate the value proposition, driving higher ticket sales and additional purchases.\n"

        "\n3. Weekend Relaxation and Wellness Packages: Develop relaxation and wellness-focused weekend packages, including stress-relief activities, spa experiences, or health-themed events\n"
        "- Wellness and relaxation activities can be a unique selling point for working adults looking for an alternative to traditional amusement park experiences.\n"
        "- These experiences could be monetized through premium pricing for spa treatments, wellness seminars, or stress-relief activities like yoga sessions or nature walks.\n"
        "- The packages can encourage longer stays, resulting in higher spending on food, merchandise, and optional add-ons.\n"
        "- Offering wellness experiences could attract affluent working adults who may be willing to pay for higher-end experiences and services, leading to increased overall revenue.\n"
        "- Marketing these packages as exclusive or limited-time offerings can drive demand and create a sense of urgency, boosting sales.\n")


  ### Conclusion ###
  print('\n\033[1m' + 'Conclusion' + '\033[0;0m\n')
  print("All these strategies listed above will help to improve visitor experience, whether be in terms of financially, enjoyability, emotionally or flexibility.\n"
        "Satisfaction rates will increase if the following strategies are to be implemented. These strategies might help to achieve USS's goal of improving satisfaction rates to higher levels and at the same time, increase its revenue due to higher demand.\n")

"""### Main"""

def main():

  ### Load data ###
  current_file_path = 'uss_survey_responses.xlsx'
  historical_file_path = 'uss_historical_reviews.xlsx'
  df = load_data(current_file_path)
  df_history = load_data(historical_file_path, is_current=False)

  # Display the first few rows of the dataset
  print("\n\033[1mInspect Survey Dataset\033[0;0m\n")
  print(df.head())

  # Get number of rows and columns of df
  num_rows, num_columns = df.shape
  print("\nNumber of Rows:", num_rows)
  print("Number of Columns:", num_columns)
  print("\n")

  # Clean df_history and display first few rows
  df_history = clean_df_history(df_history)
  print("\n\033[1mInspect Historical TripAdvisor Reviews Dataset\033[0;0m\n")
  print(df_history.head())

  # Get number of rows and columns of df_history
  num_rows_history, num_columns_history = df_history.shape
  print("\nNumber of Rows:", num_rows_history)
  print("Number of Columns:", num_columns_history)
  print("\n")


  ### Step 1: Obtain And Analyze The Demographics Of Current Visitors In USS ###
  print('****************************************************************************************************************************************************************\n')
  analyse_survey_responses(df)


  ### Step 2: Obtain And Analyze The Demographics Of Visitors Who Visited USS Some Time Ago ###
  print('****************************************************************************************************************************************************************\n')
  analyse_reviews(df_history)


  ### Step 3: Compare the Visitor Demographics + Identify Trends and Majority Classes ###
  print('****************************************************************************************************************************************************************\n')
  compare(df, df_history)


  ### Step 4: Suggest Effective Strategies Tailored To Identified Groups ###
  print('****************************************************************************************************************************************************************\n')
  suggestions()


  ### END ###
  print('****************************************************************************************************************************************************************\n')

if __name__ == "__main__":
    main()
