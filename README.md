# DSA3101 Group Project Group 3

## Enhancing Guest Experience Through Data-Driven Journey Mapping and Analysis 

#### Introduction 🎉

This project aims to analyze guest experiences at Universal Studios Singapore (USS), addressing growing concerns about increasing dissatisfaction rates among visitors. Despite efforts to enhance guest satisfaction, USS has struggled to pinpoint the key factors driving this decline, making it difficult to implement targeted improvements.

By leveraging a data-driven approach, predictive modeling and agent-based simulations, this project seeks to optimize guest flow and enhance the overall visitor experience. Poor satisfaction ratings can lead to declining attendance, reduced revenue, and ultimately lower profitability for USS. By identifying and addressing pain points, the project aims to boost operational efficiency, improve guest satisfaction and drive revenue growth, ensuring USS remains a top entertainment destination.

#### How the project came about - Problems faced by entertainment venues ⚠️

The project was initiated in response to growing challenges faced by Universal Studios Singapore (USS) in maintaining high guest satisfaction amidst evolving visitor expectations. In recent years, demographic shifts have led to a more diverse audience with varied preferences and demands, making it increasingly difficult to cater to all guest segments effectively. Additionally, guest dissatisfaction rates have been rising, with common complaints related to long wait times, overcrowding, inefficient park layouts and inconsistent service quality.

Despite USS's efforts to enhance operations, they have struggled to pinpoint the exact causes behind this growing dissatisfaction. This issue is critical because poor guest experiences can lead to negative reviews, reduced visitor retention, lower attendance, and ultimately a decline in revenue and profitability. Furthermore, the increased reliance on digital experiences and higher expectations for seamless, personalized services put additional pressure on theme parks to modernize their approach. Recognizing these challenges, this project was developed to leverage data-driven insights, predictive modeling and agent-based simulations to identify inefficiencies, optimize guest flow and enhance overall satisfaction, ultimately improving USS’s long-term operational success.

#### What are the motivations of the project? 🤩

The motivation behind this project stems from USS’s need to accurately identify and address the root causes of increasing customer dissatisfaction. Despite ongoing efforts to enhance visitor experiences, USS has struggled to pinpoint specific pain points that contribute to negative reviews, reduced visitor retention and declining profitability. By leveraging data-driven insights, the project aims to analyze satisfaction trends, identify key drivers of dissatisfaction and develop targeted solutions to improve overall guest experiences. 

Another crucial motivation is to better understand changing visitor demographics and their evolving expectations. With shifting consumer behaviors and preferences, USS must be able to segment guests based on their interests, behaviors and visit patterns, allowing for a more personalized and efficient approach to park management. Through advanced analytics and predictive modeling, this project seeks to identify market trends and develop strategic initiatives tailored to different customer segments. By optimizing guest flow, enhancing operational efficiency and designing customized experiences for various demographics, the project ultimately aims to boost visitor satisfaction, increase attendance and maximize revenue for USS in an increasingly competitive entertainment industry.

#### What problems the project hopes to solve? 💰

This project aims to conduct an in-depth analysis of the various factors influencing guest experiences at USS, both positively and negatively. By examining these aspects, the project seeks to not only identify key drivers of visitor satisfaction and dissatisfaction but also to quantify the relative impact of each factor on the overall guest journey. Understanding the significance of these factors such as wait times, attraction availability, crowd density, staff interactions, pricing, and amenities is crucial in developing effective strategies to enhance visitor experiences. More importantly, by determining the weight and influence of each contributing element, USS can make data-driven decisions to optimize park operations, improve guest engagement and implement targeted improvements. This insight will help USS prioritize enhancements that yield the highest impact on customer satisfaction, ultimately leading to a more seamless, enjoyable and memorable theme park experience.

#### What is the intended use of the project? ❓

This project will enable USS to increase guest satisfaction and retention through data-driven approaches such as analysis and modelling. 

By analysing data collected from our survey and datasets available online, we will identify key factors influencing guest satisfaction at USS. Guest journey analysis and segmentation will reveal behavioural patterns and preferences across different visitor groups, allowing for customised promotions, optimized operational workflows and deliver enhanced, personalised experiences tailored to each guest segment, therefore driving higher satisfaction and retention levels. 

Predictive modelling will allow for experience optimisation by enhancing operational decision making through forecasting attendance patterns to optimise staff allocation, simulating attraction layouts to minimise wait times and anticipate potential complaints before they occur, allowing USS to implement preventive measures or provide timely support. These capabilities will directly contribute to increased revenue and guest satisfaction. 

Furthermore, we will explore the use of IoT devices to enable real-time crowd monitoring to facilitate dynamic resource allocation, This would enhance responsiveness to unexpected situations, therefore providing USS with operational agility. 


#### Project Content 📒

| Sections                                      | Business Question To Analyze                                                                 | Strategy/Approach Used To Tackle The Section |
|-----------------------------------------------|--------------------------------------------------------------------------------------------|----------------------------------------------|
| **Key Factors Influencing Guest Satisfaction** | Which categories related to USS have the highest dissatisfaction, and what key factors should be improved to enhance guest experience? | - Conducted a survey gathering approximately 500 responses to assess overall guest satisfaction at USS through various questions  <br> <br> - Leveraged business metrics like Customer Satisfaction Score (CSAT) and Net Promoter Score (NPS) to determine overall satisfaction rates of USS, as well as various sections in USS such as Ticketing and Rides & Attractions. <br> <br> - Created bar plots to identify the most important reasons for customer unsatisfaction rates for various sections of USS and ranked mean importance of factors which affect customers' satisfaction rates <br> <br> - Identified the top 3 reasons for customers' current unsatisfaction for various sections of USS <br> <br> - Implemented appropriate and relevant strategies to address the most common pain points in customers from the top 3 reasons in order to improve customer satisfaction and as a result boost revenue and profits for USS|
| **Guest Segmentation Model**                  | What are the key guest segments and their traits?                                          | - Developed a K-means model to categorize theme park visitors into distinct clusters based on demographic traits, visit behaviors, and attraction preferences, enabling data-driven decision-making <br> <br> - Evaluated key decision-making factors influencing visitor attendance, such as ticket pricing, ride wait times, special events, and seasonal trends, to identify key visitor groups <br> <br> - Segmented guests based on spending patterns and visit frequency, various behaviours and preferences, facilitating personalized promotions and improved guest engagement |
| **Guest Journey Patterns**                    | How do common guest journey patterns provide opportunities for personalization and operational improvements? | |
| **Impact Of Marketing Strategies On Guest Behaviour** | How have marketing strategies changed guest segments and satisfaction over time? | - Utilised survey data and historical TripAdvisor reviews obtained to compare how guest segments and satisfaction have changed over time. <br> <br> - Guest segments assessed include visitor demographics (age, gender, nationality), type of visitor (solo traveller, visiting with friends, families with young children/teenagers/elderly) and preferred time of day to visit USS. <br> <br> - Identified key shifts in guest segment proportions using appropriate visualisations and analysed the importance of special events in attracting visitors. <br> <br> - Proposed relevant marketing strategies targeted towards identified groups that saw a decrease in proportion. This enables us to broaden USS’s appeal and improve the guest experience of underrepresented groups so as to enhance overall guest satisfaction. |
| **External Factors and Guest Segmentation**   | Investigate the influence of external factors on segment size and behavior. Suggest operational adjustments for high-impact periods. | |
| **Demand Prediction for Attractions and Services** | How can we predict visitor demand for attractions and services using historical trends, weather, and events to optimize resources and enhance customer experience? |- Collected and cleaned wait time data for each ride from Thrill Data and generated visitor count estimates for eateries at USS. Additionally, weather data was sourced from Meteorological Service Singapore and pre-processed for integration. <br> <br> - Employed machine learning models, including XGBoost and Random Forest, alongside SARIMA time series models to forecast demand. Model performance was evaluated using MAE, RMSE, and R². <br> <br> - Predictions were conducted at various levels, including hourly wait times for each ride, average daily wait times, and overall USS wait times. <br> <br> - Assessed the influence of external factors (e.g., weather, holidays, seasonal events) on demand through model interpretation and feature importance analysis.|
| **Optimization of Attraction Layouts and Schedules** | How can we optimize attraction layouts and schedules to enhance guest satisfaction at USS? | |
| **Resource Allocation for Demand Variability** | How can we allocate staff dynamically to meet the variability in demand of visitors throughout the day to ensure a pleasant visitor experience? | - Utilised wait time data for rides to assign popularity weights to attractions in USS <br> <br> - Utilised Agent Based Modeling to simulate the behavior of individual agents: visitors and staff members within the theme park, allowing for dynamic and adaptive resource allocation. The staff are mobilised to respond to real-time demand signals, such as number of visitors and length of wait time for rides and a rostering is then created based on predefined rules. <br> <br> - This allows for a flexible, staffing solution that optimizes resource distribution in response to demand variability, ensuring that staff are efficiently deployed where they are most needed, improving operational performance and customer satisfaction. |
| **Predicting Guest Complaints and Service Recovery** | How can USS leverage historical review data to proactively detect and address guest complaints, improving satisfaction, protecting its online reputation, and driving repeat visits? | - Developed a machine learning model to predict guest complaints by classifying reviews based on star ratings (1–2 stars as "complaints", 3–5 as "non-complaints"), using a TF-IDF vectorizer and Support Vector Machine (SVM) classifier.  <br> <br> - The model achieved 94% accuracy, with 62% precision and 64% recall for detecting complaints—demonstrating strong performance even with imbalanced data. <br> <br> - Proposed the integration of an exit survey at USS to collect real-time guest feedback at key touchpoints (e.g., ride exits, park exits). <br> <br> - Survey responses can be analyzed on the spot using the trained model to flag potential complaints for immediate follow-up or recovery action. <br> <br> - This proactive strategy helps address issues before guests leave the park and protects USS’s online reputation on review platforms like TripAdvisor, which heavily influence future visitor decisions. |
| **IoT Data Integration for Experience Optimization** | How willing are people to wear a digital watch and how feasible and impactful is IoT in a theme park? |- Conducted a comprehensive survey targeting approximately 500 park visitors to gauge their willingness to adopt IoT devices, such as digital watches, and enhance their experience at Universal Studios Singapore (USS). <br> <br> - Applied statistical methods to analyze the data, focusing on the correlation between visitors' willingness to wear digital watches and other factors such as age, tourist status (local vs. tourist), and their current satisfaction with park facilities.  <br> <br> - Utilized visual tools like bar graphs and correlation matrices to display the distribution and relationship of willingness across different demographic and experiential factors.  <br> <br> - Identified key demographic segments and park experience factors that significantly influence the willingness to use IoT devices in the park setting.  <br> <br> - Developed targeted strategies to address barriers to IoT adoption, enhancing operational decisions and potentially improving overall guest satisfaction and park efficiency through the strategic use of IoT technology.  |

#### Benefits of the project ⭐

The project offers several potential benefits for Universal Studios Singapore (USS), primarily focusing on enhancing guest satisfaction, improving operational efficiency, and boosting revenue. 

1) Improved Guest Satisfaction – By identifying the key factors that contribute to both positive and negative guest experiences, the project provides actionable insights that can help optimize guest journeys across the park. These insights can be used to improve areas such as ride experiences, wait times, staff interactions and overall park layout, leading to a more enjoyable and seamless experience for visitors.

2) Optimized Park Operations – Through the use of data-driven optimization models, the project helps USS identify operational bottlenecks, particularly around ride attractions and other high-traffic areas. This allows for better resource allocation, such as adjusting staffing levels or optimizing ride throughput, ultimately improving the efficiency of park operations and reducing unnecessary delays.

3) Targeted Customer Segmentation – By analyzing visitor demographics and behavior patterns, the project enables USS to segment its audience based on interests, preferences and behaviors. This allows for the development of personalized experiences, such as tailored marketing campaigns, targeted promotions or customized park experiences that cater to different customer segments, improving visitor retention and engagement.

4) Increased Revenue and Profitability – Improving guest satisfaction and operational efficiency directly impacts USS's bottom line. With optimized guest flow, shorter wait times and personalized offerings, visitors are likely to spend more time and money within the park, whether on rides, eateries or souvenir shops. By increasing guest spending and enhancing retention, the project can drive higher attendance rates and greater overall revenue.

5) Data-Driven Decision Making – The project empowers USS to make more informed, data-driven decisions when it comes to strategic planning and operational improvements. With a clearer understanding of guest needs, preferences and pain points, USS can prioritize investments in infrastructure, attractions and services that are most likely to enhance the guest experience and generate a strong return on investment.

#### Limitations of the project 😔

While the optimization model provided valuable insights into the impact of changes on ride attractions, there are several limitations that affect the accuracy and comprehensiveness of the results obtained. 

1) Limited Impact Assessment on Eateries and Souvenir Shops – The model effectively analyzed ride attraction efficiency, but due to the lack of comprehensive data on crowd distribution in eateries and souvenir shops, it was unable to accurately reflect wait times for these locations. As a result, the model reported little to zero wait times, which may not be an accurate representation of real-world conditions.

2) Lack of Sufficient Survey Data – The analysis relied on survey responses, but only around 500+ responses were collected, which may not be statistically representative of the full range of visitor experiences at USS. A larger sample size would improve the reliability of insights into customer satisfaction, preferences and pain points.

3) Limited Scope of Machine Learning Models – Due to data constraints, only a select number of Machine Learning models were applied to analyze trends and predict guest experiences. A wider selection of models, including deep learning techniques and reinforcement learning, could have provided more nuanced insights but were not feasible within the project's scope.

4) Use of External Datasets – Some datasets used in the analysis were sourced from Universal Studios locations outside of Singapore, which may not fully reflect USS’s unique visitor behaviors, crowd flow patterns or operational challenges. While data cleaning and preprocessing were conducted to enhance accuracy, differences in park layouts, local visitor demographics and cultural preferences may still introduce inconsistencies in the findings.

5) Challenges in Forecasting Future Visitor Trends – The project primarily focused on analyzing historical and current data, making it difficult to accurately forecast future visitor flow and satisfaction trends. Additionally, external factors such as seasonal events, economic conditions, global travel restrictions and new attraction launches were not explicitly modeled, potentially impacting the long-term applicability of the results.

#### Challenges faced during the project 🤯

Throughout the project, several challenges emerged, affecting the data collection, model implementation and overall analysis process. These challenges had to be carefully managed to ensure the accuracy and reliability of the findings.

1) Survey Response Accuracy and Bias – One of the major challenges was the potential inaccuracy of survey responses. Since there were no incentives for completing the survey, some respondents might have rushed through or provided random answers, reducing the reliability of the data. Additionally, majority-minority bias was a concern, where responses could be skewed towards certain categories, making it difficult to obtain a balanced representation of guest experiences at Universal Studios Singapore (USS).

2) Difficulty in Collecting Sufficient Responses – Gathering enough survey responses for model implementation was a time-consuming and labor-intensive task. The team had to manually reach out to friends, family, passersby and online survey platforms, which required significant effort. Due to the limited number of responses, data augmentation techniques such as imputation were applied to increase dataset size, but this introduced a risk of reducing the authenticity of the data.

3) Selecting the Best Machine Learning Model – Another key challenge was determining the most suitable Machine Learning model for analyzing guest satisfaction and predicting visitor flow. This required an exhaustive cost-benefit analysis of different models, balancing factors such as accuracy, evaluation metrics (precision, recall, F1-score), efficiency, model complexity and risk of overfitting. Some models performed well in terms of accuracy but were computationally expensive, while others were efficient but had weaker predictive performance, making it difficult to decide on an optimal approach.

#### Conclusion 📈

This project provided valuable insights into guest experiences at USS by leveraging data-driven analysis, predictive modeling and optimization techniques. Through the identification of key dissatisfaction factors, guest segmentation and demand forecasting, we were able to pinpoint critical areas for improvement including long wait times, inefficient crowd management and gaps in service quality.

Despite limitations such as survey response biases, reliance on external datasets and forecasting challenges, our findings offer actionable recommendations to enhance visitor satisfaction. By implementing targeted strategies—such as optimizing ride schedules, personalizing guest experiences and dynamically allocating resources — USS can improve operational efficiency, elevate guest engagement and ultimately drive higher revenue and profitability. Moving forward, future iterations of this project could integrate real-time IoT data, incorporate more advanced machine learning models and refine predictive analytics to further enhance decision-making. With continuous improvements, USS can stay ahead of evolving guest expectations and maintain its reputation as a top-tier entertainment destination.

#### GitHub Repository Navigation 🧑‍💻


Explanation of Directories and Files:

- **Notebooks/**: Contains Google Colab Notebooks for analysis and experiments.
  - **Subgroup A/**: Notebooks specific to subgroup A.
  - **Subgroup B/**: Notebooks specific to subgroup B.
  
- **data/**: Contains datasets used in the project.
  - **Raw/**: Raw, unprocessed data (survey data)
  - **External/**: External datasets from other sources (online, internet)
  - **Processed/**: Cleaned and processed data
  
- **src/**: Contains source code for the project.
  - **Subgroup A/**: Source code for subgroup A (in python)
  - **Subgroup B/**: Source code for subgroup B (in python)
  
- **.gitignore**: Git ignore file to exclude unnecessary files from version control.

- **Dockerfile**: Defines the environment and dependencies for the project using Docker.

- **README.md**: This file with information about the project.

- **requirements.txt**: Lists the required Python packages for the project.

#### Subgroup Members 👯

| **Subgroup A**              | GitHub Profile                                      |
|-----------------------------|-----------------------------------------------------|
| Yu Zifan                    | [gracoco123](https://github.com/gracoco123)        |
| Low Shi Ya Amelia           | [am3lia-low](https://github.com/am3lia-low)        |
| Tan Teck Hwe Damaen         | [tadamaen](https://github.com/tadamaen)            |
| Isabella Chong              | [isabellachong](https://github.com/isabellachong)  |
| Su Xiangling, Brenda        | [brenda-su](https://github.com/brenda-su)          |

| **Subgroup B**              | GitHub Profile                                      |
|-----------------------------|-----------------------------------------------------|
| Tan Jone Lung, Jayden       | [jaydentjl2002](https://github.com/jaydentjl2002)  |
| Beh Wan Ting                | [wwwanting](https://github.com/wwwanting)          |
| Fun Wen Xin                 | [funwenxin](https://github.com/funwenxin)          |
| Yeo Jing Wen, Cheryl        | [cherylyeov](https://github.com/cherylyeov)        |
| Lien Wei Hao, Jordan        | [lienweihao](https://github.com/lienweihao)        |

