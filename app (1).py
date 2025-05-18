#!/usr/bin/env python
# coding: utf-8

# In[128]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import streamlit as st
import pytz
import datetime




# In[2]:


nltk.download('vader_lexicon')


# In[3]:


apps_df=pd.read_csv('Play Store Data.csv')
reviews_df=pd.read_csv('User Reviews.csv')


# In[6]:


#Step 2 : Data Cleaning
apps_df = apps_df.dropna(subset=['Rating'])
for column in apps_df.columns :
    apps_df[column].fillna(apps_df[column].mode()[0],inplace=True)
apps_df.drop_duplicates(inplace=True)
apps_df=apps_df=apps_df[apps_df['Rating']<=5]
reviews_df.dropna(subset=['Translated_Review'],inplace=True)


# In[7]:


#Convert the Installs columns to numeric by removing commas and +
apps_df['Installs']=apps_df['Installs'].str.replace(',','').str.replace('+','').astype(int)

#Convert Price column to numeric after removing $
apps_df['Price']=apps_df['Price'].str.replace('$','').astype(float)


# In[8]:


merged_df=pd.merge(apps_df,reviews_df,on='App',how='inner')


# In[9]:


def convert_size(size):
    if 'M' in size:
        return float(size.replace('M',''))
    elif 'k' in size:
        return float(size.replace('k',''))/1024
    else:
        return np.nan
apps_df['Size']=apps_df['Size'].apply(convert_size)


# In[10]:


#Lograrithmic
apps_df['Log_Installs']=np.log(apps_df['Installs'])


# In[11]:


apps_df['Reviews']=apps_df['Reviews'].astype(int)


# In[12]:


apps_df['Log_Reviews']=np.log(apps_df['Reviews'])


# In[13]:


def rating_group(rating):
    if rating >= 4:
        return 'Top rated app'
    elif rating >=3:
        return 'Above average'
    elif rating >=2:
        return 'Average'
    else:
        return 'Below Average'
apps_df['Rating_Group']=apps_df['Rating'].apply(rating_group)


# In[14]:


#Revenue column
apps_df['Revenue']=apps_df['Price']*apps_df['Installs']


# In[15]:


sia = SentimentIntensityAnalyzer()


# In[16]:


reviews_df['Sentiment_Score']=reviews_df['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])


# In[17]:


apps_df['Last Updated']=pd.to_datetime(apps_df['Last Updated'],errors='coerce')


# In[18]:


apps_df['Year']=apps_df['Last Updated'].dt.year

# In[22]:

plot_width=400
plot_height=300
plot_bg_color='black'
text_color='white'
title_font={'size':16}
axis_font={'size':12}


# In[23]:


#Figure 1
st.subheader("Top 10 App Categories")

category_counts=apps_df['Category'].value_counts().nlargest(10)

fig1=px.bar(
    x=category_counts.index,
    y=category_counts.values,
    labels={'x':'Category','y':'Count'},
    title='Top Categories on Play Store',
    color=category_counts.index,
    color_discrete_sequence=px.colors.sequential.Plasma,
    width=400,
    height=300
)
fig1.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)

st.plotly_chart(fig1, use_container_width=True)


# In[24]:


#Figure 2
st.subheader("Distribution of App by amount")

type_counts=apps_df['Type'].value_counts()
fig2=px.pie(
    values=type_counts.values,
    names=type_counts.index,
    title='App Type Distribution',
    color_discrete_sequence=px.colors.sequential.RdBu,
    width=400,
    height=300
)
fig2.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    margin=dict(l=10,r=10,t=30,b=10)
)

st.plotly_chart(fig2, use_container_width=True)


# In[25]:


#Figure 3
st.subheader("Rating distribution")

fig3=px.histogram(
    apps_df,
    x='Rating',
    nbins=20,
    title='Rating Distribution',
    color_discrete_sequence=['#636EFA'],
    width=400,
    height=300
)
fig3.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)

st.plotly_chart(fig3, use_container_width=True)

# In[26]:

#Figure 4
st.subheader("Sentiment distribution")

sentiment_counts=reviews_df['Sentiment_Score'].value_counts()
fig4=px.bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    labels={'x':'Sentiment Score','y':'Count'},
    title='Sentiment Distribution',
    color=sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.RdPu,
    width=400,
    height=300
)
fig4.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)

st.plotly_chart(fig4, use_container_width=True)


# In[27]:


#Figure 5
st.subheader("Installs by category")

installs_by_category=apps_df.groupby('Category')['Installs'].sum().nlargest(10)
fig5=px.bar(
    x=installs_by_category.index,
    y=installs_by_category.values,
    orientation='h',
    labels={'x':'Installs','y':'Category'},
    title='Installs by Category',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Blues,
    width=400,
    height=300
)
fig5.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)

st.plotly_chart(fig5, use_container_width=True)


# In[28]:


# Updates Per Year Plot
st.subheader("No. of updates over time")

updates_per_year = apps_df['Last Updated'].dt.year.value_counts().sort_index()
fig6 = px.line(
    x=updates_per_year.index,
    y=updates_per_year.values,
    labels={'x': 'Year', 'y': 'Number of Updates'},
    title='Number of Updates Over the Years',
    color_discrete_sequence=['#AB63FA'],
    width=plot_width,
    height=plot_height
)
fig6.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)

st.plotly_chart(fig6, use_container_width=True)


# In[29]:


#Figure 7
st.subheader("Revenue by category")

revenue_by_category=apps_df.groupby('Category')['Revenue'].sum().nlargest(10)
fig7=px.bar(
    x=installs_by_category.index,
    y=installs_by_category.values,
    labels={'x':'Category','y':'Revenue'},
    title='Revenue by Category',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Greens,
    width=400,
    height=300
)
fig7.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)

st.plotly_chart(fig7, use_container_width=True)


# In[30]:


#Figure 8
st.subheader("Top App Genres")

genre_counts=apps_df['Genres'].str.split(';',expand=True).stack().value_counts().nlargest(10)
fig8=px.bar(
    x=genre_counts.index,
    y=genre_counts.values,
    labels={'x':'Genre','y':'Count'},
    title='Top Genres',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.OrRd,
    width=400,
    height=300
)
fig8.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)

st.plotly_chart(fig8, use_container_width=True)


# In[31]:


#Figure 9
st.subheader("Impact of last update on ratings")

fig9=px.scatter(
    apps_df,
    x='Last Updated',
    y='Rating',
    color='Type',
    title='Impact of Last Update on Rating',
    color_discrete_sequence=px.colors.qualitative.Vivid,
    width=400,
    height=300
)
fig9.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)

st.plotly_chart(fig9, use_container_width=True)


# In[32]:


#Figure 10
st.subheader("Rating of paid vs free apps")

fig10=px.box(
    apps_df,
    x='Type',
    y='Rating',
    color='Type',
    title='Rating for Paid vs Free Apps',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    width=400,
    height=300
)
fig10.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)

st.plotly_chart(fig10, use_container_width=True)


# ### Task-1 : Scatter-plot to visualize the relationship between revenue and the number of installs for paid apps only

# In[34]:


paid_df=apps_df[apps_df["Price"]>0]


# In[35]:


#Figure 11
st.subheader("Revenue by no. of installs(paid apps)")

fig11=px.scatter(
    paid_df,
    x='Revenue',
    y='Installs',
    color='Type',
    title='Revenue v/s no. of installs of paid apps',
    color_discrete_sequence=px.colors.qualitative.Vivid,
    width=400,
    height=300
)
fig11.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)

st.plotly_chart(fig11, use_container_width=True)


# ### Task-2: Use a grouped bar chart to compare the average rating and total review count for the top 10 app categories by number of installs. Filter out any categories where the average rating is below 4.0 and size below 10 M and last update should be Jan month . this graph should work only between 3PM IST to 5 PM IST apart from that time we should not show this graph in dashboard itself.

# In[37]:


df2=apps_df.copy()


# In[38]:


df2['Month'] = df2['Last Updated'].dt.month_name()


# <b>Creating filtered data

# In[40]:


df2 = df2[
    (df2['Rating'] >= 4.0) &
    (df2['Size'] >= 10) &
    (df2['Month'] == 'January')
]


# In[41]:


df_grouped = df2.groupby('Category').agg(
        Avg_Rating=('Rating', 'mean'),
        Total_Reviews=('Reviews', 'sum'),
        Total_Installs=('Installs', 'sum')
    ).sort_values(by='Total_Installs', ascending=False).head(10)


# In[42]:


df_bar=df_grouped[["Avg_Rating","Total_Reviews"]]


# <b> Creating function to write messege in html

# In[44]:


# <b> Creating timestamped grouped bar chart

# In[47]:


df_bar.index.name = 'Category'
df_bar = df_bar.reset_index()


# In[48]:


melted = df_bar.melt(id_vars='Category', var_name='Metric', value_name='Value')


# In[49]:


melted.loc[melted['Metric'] == 'Total_Reviews', 'Value'] /= 1000000


# <b>Plotting graph:

# In[51]:


ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.datetime.now(ist)

if 15 <= now_ist.hour < 17:
        # Figure 12
        st.subheader("Average rating vs No. of reviews")

        fig12 = px.bar(
            melted,
            x='Category',
            y='Value',
            color='Metric',
            barmode='group',
            title='Avg Rating vs Total Reviews(in Millions) (Top Categories - Jan)',
            color_discrete_sequence=px.colors.qualitative.Vivid,
            width=400,
            height=300
        )
        fig12.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white',
            title_font={'size': 16},
            xaxis=dict(title_font={'size': 12}),
            yaxis=dict(title_font={'size': 12}),
            margin=dict(l=10, r=10, t=30, b=10)
        )

        st.plotly_chart(fig12, use_container_width=True)

else:
    st.warning("This dashboard is only accessible from 3 PM to 5 PM IST. Please come back later.")


# ### Task-3: Plot a time series line chart to show the trend of total installs over time, segmented by app category. Highlight periods of significant growth by shading the areas under the curve where the increase in installs exceeds 20% month-over-month and app name should not starts with x, y ,z and app category should start with letter " E " or " C " or " B " and We have to translate the Beauty category in Hindi and Business category in Tamil and Dating category in German while showing it on Graph. reviews should be more than 500 the app name should not contain letter "S" as well as this graph should work only between 6 PM IST to 9 PM IST apart from that time we should not show this graph in dashboard itself

# In[53]:


df3=apps_df.copy()


# <b> Creating filtered data

# In[55]:


df3 = df3[
    (~df3['App'].str[0].isin(['X', 'Y', 'Z'])) &  
    (df3['Category'].str[0].isin(['E', 'C', 'B'])) &  
    (df3['Reviews'] > 500) & 
    (~df3['App'].str.contains('S', case=False))
]


# <b>Translating app names

# In[57]:


# <b>Translating app names simplified

lang_code_map = {
    'BEAUTY': 'सौंदर्य',
    'BUSINESS': 'வணிகம்',
    'DATING': 'Dating',
}

def translate_category_nllb(category):
    return lang_code_map.get(category, category)

df3['Translated_Category'] = df3['Category'].apply(translate_category_nllb)


df3 = df3.sort_values('Last Updated')

grouped = df3.groupby(
        ['Translated_Category', pd.Grouper(key='Last Updated', freq='M')]
    )['Installs'].sum().reset_index()

grouped['pct_change'] = grouped.groupby('Translated_Category')['Installs'].pct_change()
grouped['high_growth'] = grouped['pct_change'] > 0.2
grouped['month'] = grouped['Last Updated'].dt.strftime('%Y-%m')

melted = grouped.rename(columns={
    'Translated_Category': 'Category',
    'Installs': 'Value'
    })

melted['Metric'] = melted['high_growth'].apply(lambda x: 'High Growth' if x else 'Normal Growth')


# <b>Plotting graph:

# In[62]:


import plotly.graph_objects as go

ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.datetime.now(ist)

if 18 <= now_ist.hour < 21:

    # Figure 13
    st.subheader("Monthly installs per category")

    fig13 = px.line(
        melted,
        x='month',
        y='Value',
        color='Category',
        line_group='Category',
        title='Monthly Installs per Category (Translated Names)',
        color_discrete_sequence=px.colors.qualitative.Set3,
        width=1000,
        height=500,
        hover_data=['Metric']
    )

    color_map = {}
    for trace in fig13.data:
        category = trace.name
        color = trace.line.color
        color_map[category] = color  # Store the color used by each line

    high_growth_data = melted[melted['Metric'] == 'High Growth']
    categories = high_growth_data['Category'].unique()

    for category in categories:
        category_data = high_growth_data[high_growth_data['Category'] == category]
        base_color = color_map.get(category, 'rgba(255,255,255,1)')

        if base_color.startswith('#'):
            hex_color = base_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            rgba_color = f'rgba({r}, {g}, {b}, 0.3)'
        else:
            rgba_color = base_color.replace('1)', '0.3)') if 'rgba' in base_color else base_color

        fig13.add_trace(go.Scatter(
            x=category_data['month'],
            y=category_data['Value'],
            mode='lines',
            name=f'{category} - High Growth',
            fill='tozeroy',
            line=dict(color=rgba_color),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig13.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        title_font={'size': 16},
        xaxis=dict(title='Month', title_font={'size': 12}),
        yaxis=dict(title='Total Installs', title_font={'size': 12}),
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.plotly_chart(fig13, use_container_width=True)
else:
    st.warning("This dashboard is only accessible from 6 PM to 9 PM IST. Please come back later.")

