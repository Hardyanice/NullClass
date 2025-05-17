{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 128,
   "id": "2f7a06ac-2af8-4a02-a143-1a2c1d6b05b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import plotly.express as px\n",
    "import plotly.io as pio\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n",
    "from nltk.sentiment.vader import SentimentIntensityAnalyzer\n",
    "import nltk\n",
    "import streamlit as st"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1ef7399f-116b-4307-9f6b-9a1c7a7500ff",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package vader_lexicon to\n",
      "[nltk_data]     C:\\Users\\Souhar\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package vader_lexicon is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nltk.download('vader_lexicon')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "2993793d-d367-4270-9dc0-5454ec52aa08",
   "metadata": {},
   "outputs": [],
   "source": [
    "apps_df=pd.read_csv('Play Store Data.csv')\n",
    "reviews_df=pd.read_csv('User Reviews.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "985a2547-fe8a-455b-a31d-6add57730c6d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Souhar\\AppData\\Local\\Temp\\ipykernel_44896\\2487222361.py:4: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  apps_df[column].fillna(apps_df[column].mode()[0],inplace=True)\n"
     ]
    }
   ],
   "source": [
    "#Step 2 : Data Cleaning\n",
    "apps_df = apps_df.dropna(subset=['Rating'])\n",
    "for column in apps_df.columns :\n",
    "    apps_df[column].fillna(apps_df[column].mode()[0],inplace=True)\n",
    "apps_df.drop_duplicates(inplace=True)\n",
    "apps_df=apps_df=apps_df[apps_df['Rating']<=5]\n",
    "reviews_df.dropna(subset=['Translated_Review'],inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "212988f8-1186-4abc-8822-c165dac916b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Convert the Installs columns to numeric by removing commas and +\n",
    "apps_df['Installs']=apps_df['Installs'].str.replace(',','').str.replace('+','').astype(int)\n",
    "\n",
    "#Convert Price column to numeric after removing $\n",
    "apps_df['Price']=apps_df['Price'].str.replace('$','').astype(float)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "1cbb6e81-96db-4ff2-b370-216d194e2624",
   "metadata": {},
   "outputs": [],
   "source": [
    "merged_df=pd.merge(apps_df,reviews_df,on='App',how='inner')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "04ab5180-3733-4bdb-9061-33bbf4ec0118",
   "metadata": {},
   "outputs": [],
   "source": [
    "def convert_size(size):\n",
    "    if 'M' in size:\n",
    "        return float(size.replace('M',''))\n",
    "    elif 'k' in size:\n",
    "        return float(size.replace('k',''))/1024\n",
    "    else:\n",
    "        return np.nan\n",
    "apps_df['Size']=apps_df['Size'].apply(convert_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "798cfb6b-15a3-467a-8a8b-14e164d8c05d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Lograrithmic\n",
    "apps_df['Log_Installs']=np.log(apps_df['Installs'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "fd5b230d-c359-4f2a-9ae4-0439ca5d569b",
   "metadata": {},
   "outputs": [],
   "source": [
    "apps_df['Reviews']=apps_df['Reviews'].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "c4aae325-9932-44fd-b2ff-53345742f265",
   "metadata": {},
   "outputs": [],
   "source": [
    "apps_df['Log_Reviews']=np.log(apps_df['Reviews'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "14851fde-6a49-4801-9323-01844f86c2de",
   "metadata": {},
   "outputs": [],
   "source": [
    "def rating_group(rating):\n",
    "    if rating >= 4:\n",
    "        return 'Top rated app'\n",
    "    elif rating >=3:\n",
    "        return 'Above average'\n",
    "    elif rating >=2:\n",
    "        return 'Average'\n",
    "    else:\n",
    "        return 'Below Average'\n",
    "apps_df['Rating_Group']=apps_df['Rating'].apply(rating_group)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "66abb408-414c-4405-b10e-1934eeb84499",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Revenue column\n",
    "apps_df['Revenue']=apps_df['Price']*apps_df['Installs']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "7c108f65-f43c-47c9-9bbf-a490ea42507e",
   "metadata": {},
   "outputs": [],
   "source": [
    "sia = SentimentIntensityAnalyzer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "15075238-a076-4ced-abb6-63f9a0dac6bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "reviews_df['Sentiment_Score']=reviews_df['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "47eec6b1-a2a5-4847-8d04-85099f3e7363",
   "metadata": {},
   "outputs": [],
   "source": [
    "apps_df['Last Updated']=pd.to_datetime(apps_df['Last Updated'],errors='coerce')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "13ea06d7-b258-49e6-bda0-85665598d782",
   "metadata": {},
   "outputs": [],
   "source": [
    "apps_df['Year']=apps_df['Last Updated'].dt.year"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "9ca1587b-6854-4409-8a7e-61a440f2acd0",
   "metadata": {},
   "outputs": [],
   "source": [
    "html_files_path=\"./\"\n",
    "if not os.path.exists(html_files_path):\n",
    "    os.makedirs(html_files_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "f21538f4-5121-4d53-a45b-f0b8bc08029a",
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_containers=\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "b7d0dc2b-29b0-4513-ba68-f80402c57e9c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save each Plotly figure to an HTML file\n",
    "def save_plot_as_html(fig, filename, insight):\n",
    "    global plot_containers\n",
    "    filepath = os.path.join(html_files_path, filename)\n",
    "    html_content = pio.to_html(fig, full_html=False, include_plotlyjs='inline')\n",
    "    # Append the plot and its insight to plot_containers\n",
    "    plot_containers += f\"\"\"\n",
    "    <div class=\"plot-container\" id=\"{filename}\" onclick=\"openPlot('{filename}')\">\n",
    "        <div class=\"plot\">{html_content}</div>\n",
    "        <div class=\"insights\">{insight}</div>\n",
    "    </div>\n",
    "    \"\"\"\n",
    "    fig.write_html(filepath, full_html=False, include_plotlyjs='inline')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "fbf0b110-38c4-4e3d-b076-1ce6719e56d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_width=400\n",
    "plot_height=300\n",
    "plot_bg_color='black'\n",
    "text_color='white'\n",
    "title_font={'size':16}\n",
    "axis_font={'size':12}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "9cf5b5a9-c552-4e0d-a8df-5531e6462e35",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Figure 1\n",
    "st.subheader(\"Top 10 App Categories\")\n",
    "\n",
    "category_counts=apps_df['Category'].value_counts().nlargest(10)\n",
    "\n",
    "fig1=px.bar(\n",
    "    x=category_counts.index,\n",
    "    y=category_counts.values,\n",
    "    labels={'x':'Category','y':'Count'},\n",
    "    title='Top Categories on Play Store',\n",
    "    color=category_counts.index,\n",
    "    color_discrete_sequence=px.colors.sequential.Plasma,\n",
    "    width=400,\n",
    "    height=300\n",
    ")\n",
    "fig1.update_layout(\n",
    "    plot_bgcolor='black',\n",
    "    paper_bgcolor='black',\n",
    "    font_color='white',\n",
    "    title_font={'size':16},\n",
    "    xaxis=dict(title_font={'size':12}),\n",
    "    yaxis=dict(title_font={'size':12}),\n",
    "    margin=dict(l=10,r=10,t=30,b=10)\n",
    ")\n",
    "\n",
    "st.plotly_chart(fig1, use_container_width=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "e8f6d8cd-b34a-4b39-b11d-fc404a973e47",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Figure 2\n",
    "st.subheader(\"Distribution of App by amount\")\n",
    "\n",
    "type_counts=apps_df['Type'].value_counts()\n",
    "fig2=px.pie(\n",
    "    values=type_counts.values,\n",
    "    names=type_counts.index,\n",
    "    title='App Type Distribution',\n",
    "    color_discrete_sequence=px.colors.sequential.RdBu,\n",
    "    width=400,\n",
    "    height=300\n",
    ")\n",
    "fig2.update_layout(\n",
    "    plot_bgcolor='black',\n",
    "    paper_bgcolor='black',\n",
    "    font_color='white',\n",
    "    title_font={'size':16},\n",
    "    margin=dict(l=10,r=10,t=30,b=10)\n",
    ")\n",
    "\n",
    "st.plotly_chart(fig2, use_container_width=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "b9700afd-f34d-4c0b-bc71-a2909f17d4a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Figure 3\n",
    "st.subheader(\"Rating distribution\")\n",
    "\n",
    "fig3=px.histogram(\n",
    "    apps_df,\n",
    "    x='Rating',\n",
    "    nbins=20,\n",
    "    title='Rating Distribution',\n",
    "    color_discrete_sequence=['#636EFA'],\n",
    "    width=400,\n",
    "    height=300\n",
    ")\n",
    "fig3.update_layout(\n",
    "    plot_bgcolor='black',\n",
    "    paper_bgcolor='black',\n",
    "    font_color='white',\n",
    "    title_font={'size':16},\n",
    "    xaxis=dict(title_font={'size':12}),\n",
    "    yaxis=dict(title_font={'size':12}),\n",
    "    margin=dict(l=10,r=10,t=30,b=10)\n",
    ")\n",
    "\n",
    "st.plotly_chart(fig3, use_container_width=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "d801fa0c-c13c-4e69-aca1-349947857bc0",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Figure 4\n",
    "st.subheader(\"Sentiment distribution\")\n",
    "\n",
    "sentiment_counts=reviews_df['Sentiment_Score'].value_counts()\n",
    "fig4=px.bar(\n",
    "    x=sentiment_counts.index,\n",
    "    y=sentiment_counts.values,\n",
    "    labels={'x':'Sentiment Score','y':'Count'},\n",
    "    title='Sentiment Distribution',\n",
    "    color=sentiment_counts.index,\n",
    "    color_discrete_sequence=px.colors.sequential.RdPu,\n",
    "    width=400,\n",
    "    height=300\n",
    ")\n",
    "fig4.update_layout(\n",
    "    plot_bgcolor='black',\n",
    "    paper_bgcolor='black',\n",
    "    font_color='white',\n",
    "    title_font={'size':16},\n",
    "    xaxis=dict(title_font={'size':12}),\n",
    "    yaxis=dict(title_font={'size':12}),\n",
    "    margin=dict(l=10,r=10,t=30,b=10)\n",
    ")\n",
    "\n",
    "st.plotly_chart(fig4, use_container_width=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "6cddb871-554e-45f4-9f62-48a3b3862a56",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Figure 5\n",
    "st.subheader(\"Installs by category\")\n",
    "\n",
    "installs_by_category=apps_df.groupby('Category')['Installs'].sum().nlargest(10)\n",
    "fig5=px.bar(\n",
    "    x=installs_by_category.index,\n",
    "    y=installs_by_category.values,\n",
    "    orientation='h',\n",
    "    labels={'x':'Installs','y':'Category'},\n",
    "    title='Installs by Category',\n",
    "    color=installs_by_category.index,\n",
    "    color_discrete_sequence=px.colors.sequential.Blues,\n",
    "    width=400,\n",
    "    height=300\n",
    ")\n",
    "fig5.update_layout(\n",
    "    plot_bgcolor='black',\n",
    "    paper_bgcolor='black',\n",
    "    font_color='white',\n",
    "    title_font={'size':16},\n",
    "    xaxis=dict(title_font={'size':12}),\n",
    "    yaxis=dict(title_font={'size':12}),\n",
    "    margin=dict(l=10,r=10,t=30,b=10)\n",
    ")\n",
    "\n",
    "st.plotly_chart(fig5, use_container_width=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "d4d609ee-4e8a-42d9-aa2a-39d784574bcc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Updates Per Year Plot\n",
    "st.subheader(\"No. of updates over time\")\n",
    "\n",
    "updates_per_year = apps_df['Last Updated'].dt.year.value_counts().sort_index()\n",
    "fig6 = px.line(\n",
    "    x=updates_per_year.index,\n",
    "    y=updates_per_year.values,\n",
    "    labels={'x': 'Year', 'y': 'Number of Updates'},\n",
    "    title='Number of Updates Over the Years',\n",
    "    color_discrete_sequence=['#AB63FA'],\n",
    "    width=plot_width,\n",
    "    height=plot_height\n",
    ")\n",
    "fig6.update_layout(\n",
    "    plot_bgcolor=plot_bg_color,\n",
    "    paper_bgcolor=plot_bg_color,\n",
    "    font_color=text_color,\n",
    "    title_font=title_font,\n",
    "    xaxis=dict(title_font=axis_font),\n",
    "    yaxis=dict(title_font=axis_font),\n",
    "    margin=dict(l=10, r=10, t=30, b=10)\n",
    ")\n",
    "\n",
    "st.plotly_chart(fig6, use_container_width=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "b5068c5f-7591-47a3-b2c7-29edeabdb875",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Figure 7\n",
    "st.subheader(\"Revenue by category\")\n",
    "\n",
    "revenue_by_category=apps_df.groupby('Category')['Revenue'].sum().nlargest(10)\n",
    "fig7=px.bar(\n",
    "    x=installs_by_category.index,\n",
    "    y=installs_by_category.values,\n",
    "    labels={'x':'Category','y':'Revenue'},\n",
    "    title='Revenue by Category',\n",
    "    color=installs_by_category.index,\n",
    "    color_discrete_sequence=px.colors.sequential.Greens,\n",
    "    width=400,\n",
    "    height=300\n",
    ")\n",
    "fig7.update_layout(\n",
    "    plot_bgcolor='black',\n",
    "    paper_bgcolor='black',\n",
    "    font_color='white',\n",
    "    title_font={'size':16},\n",
    "    xaxis=dict(title_font={'size':12}),\n",
    "    yaxis=dict(title_font={'size':12}),\n",
    "    margin=dict(l=10,r=10,t=30,b=10)\n",
    ")\n",
    "\n",
    "st.plotly_chart(fig7, use_container_width=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "07bde163-7d5e-4442-8aa4-6a12d474a6d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Figure 8\n",
    "st.subheader(\"Top App Genres\")\n",
    "\n",
    "genre_counts=apps_df['Genres'].str.split(';',expand=True).stack().value_counts().nlargest(10)\n",
    "fig8=px.bar(\n",
    "    x=genre_counts.index,\n",
    "    y=genre_counts.values,\n",
    "    labels={'x':'Genre','y':'Count'},\n",
    "    title='Top Genres',\n",
    "    color=installs_by_category.index,\n",
    "    color_discrete_sequence=px.colors.sequential.OrRd,\n",
    "    width=400,\n",
    "    height=300\n",
    ")\n",
    "fig8.update_layout(\n",
    "    plot_bgcolor='black',\n",
    "    paper_bgcolor='black',\n",
    "    font_color='white',\n",
    "    title_font={'size':16},\n",
    "    xaxis=dict(title_font={'size':12}),\n",
    "    yaxis=dict(title_font={'size':12}),\n",
    "    margin=dict(l=10,r=10,t=30,b=10)\n",
    ")\n",
    "\n",
    "st.plotly_chart(fig8, use_container_width=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "1b24ce25-f0d3-483e-bad8-8b3929eae642",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Figure 9\n",
    "st.subheader(\"Impact of last update on ratings\")\n",
    "\n",
    "fig9=px.scatter(\n",
    "    apps_df,\n",
    "    x='Last Updated',\n",
    "    y='Rating',\n",
    "    color='Type',\n",
    "    title='Impact of Last Update on Rating',\n",
    "    color_discrete_sequence=px.colors.qualitative.Vivid,\n",
    "    width=400,\n",
    "    height=300\n",
    ")\n",
    "fig9.update_layout(\n",
    "    plot_bgcolor='black',\n",
    "    paper_bgcolor='black',\n",
    "    font_color='white',\n",
    "    title_font={'size':16},\n",
    "    xaxis=dict(title_font={'size':12}),\n",
    "    yaxis=dict(title_font={'size':12}),\n",
    "    margin=dict(l=10,r=10,t=30,b=10)\n",
    ")\n",
    "\n",
    "st.plotly_chart(fig9, use_container_width=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "617af2ab-b2dd-4437-b9cd-d86cfa24b61b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Figure 10\n",
    "st.subheader(\"Rating of paid vs free apps\")\n",
    "\n",
    "fig10=px.box(\n",
    "    apps_df,\n",
    "    x='Type',\n",
    "    y='Rating',\n",
    "    color='Type',\n",
    "    title='Rating for Paid vs Free Apps',\n",
    "    color_discrete_sequence=px.colors.qualitative.Pastel,\n",
    "    width=400,\n",
    "    height=300\n",
    ")\n",
    "fig10.update_layout(\n",
    "    plot_bgcolor='black',\n",
    "    paper_bgcolor='black',\n",
    "    font_color='white',\n",
    "    title_font={'size':16},\n",
    "    xaxis=dict(title_font={'size':12}),\n",
    "    yaxis=dict(title_font={'size':12}),\n",
    "    margin=dict(l=10,r=10,t=30,b=10)\n",
    ")\n",
    "\n",
    "st.plotly_chart(fig10, use_container_width=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "efeefec2-e5dd-495e-be5e-7d5b638486d1",
   "metadata": {},
   "source": [
    "### Task-1 : Scatter-plot to visualize the relationship between revenue and the number of installs for paid apps only"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "13ba3ded-4487-435b-b7b4-9d3bd357848b",
   "metadata": {},
   "outputs": [],
   "source": [
    "paid_df=apps_df[apps_df[\"Price\"]>0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "0004e7b7-63b2-4c8f-9eea-4c22bf71b1f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Figure 11\n",
    "st.subheader(\"Revenue by no. of installs(paid apps)\")\n",
    "\n",
    "fig11=px.scatter(\n",
    "    paid_df,\n",
    "    x='Revenue',\n",
    "    y='Installs',\n",
    "    color='Type',\n",
    "    title='Revenue v/s no. of installs of paid apps',\n",
    "    color_discrete_sequence=px.colors.qualitative.Vivid,\n",
    "    width=400,\n",
    "    height=300\n",
    ")\n",
    "fig11.update_layout(\n",
    "    plot_bgcolor='black',\n",
    "    paper_bgcolor='black',\n",
    "    font_color='white',\n",
    "    title_font={'size':16},\n",
    "    xaxis=dict(title_font={'size':12}),\n",
    "    yaxis=dict(title_font={'size':12}),\n",
    "    margin=dict(l=10,r=10,t=30,b=10)\n",
    ")\n",
    "\n",
    "st.plotly_chart(fig11, use_container_width=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4e5b8ee5-8d06-4169-9b90-fbb47a0198a6",
   "metadata": {},
   "source": [
    "### Task-2: Use a grouped bar chart to compare the average rating and total review count for the top 10 app categories by number of installs. Filter out any categories where the average rating is below 4.0 and size below 10 M and last update should be Jan month . this graph should work only between 3PM IST to 5 PM IST apart from that time we should not show this graph in dashboard itself."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "07f2e1f2-18ce-4098-bcfa-b19fbeeff685",
   "metadata": {},
   "outputs": [],
   "source": [
    "df2=apps_df.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "b2a6f9e1-f95b-49bb-9443-e384812381fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "df2['Month'] = df2['Last Updated'].dt.month_name()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "32638126-6525-4186-a060-b71d1ab95a9f",
   "metadata": {},
   "source": [
    "<b>Creating filtered data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "8875847b-d1a6-4b51-8000-dc45b941fdf3",
   "metadata": {},
   "outputs": [],
   "source": [
    "df2 = df2[\n",
    "    (df2['Rating'] >= 4.0) &\n",
    "    (df2['Size'] >= 10) &\n",
    "    (df2['Month'] == 'January')\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "56a11b97-0a02-4deb-8f77-0b8e2d71672a",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_grouped = df2.groupby('Category').agg(\n",
    "        Avg_Rating=('Rating', 'mean'),\n",
    "        Total_Reviews=('Reviews', 'sum'),\n",
    "        Total_Installs=('Installs', 'sum')\n",
    "    ).sort_values(by='Total_Installs', ascending=False).head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "341f31e8-d2c1-4921-bb0a-155d417e06a0",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_bar=df_grouped[[\"Avg_Rating\",\"Total_Reviews\"]]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "19b7d0fd-0972-478f-b838-1e4cb512d253",
   "metadata": {},
   "source": [
    "<b> Creating function to write messege in html"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "1e3fe242-306d-459c-b88e-bbb3f72fb88a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pytz\n",
    "import datetime"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4d54ac43-f892-40c5-a728-04fa2fd82b76",
   "metadata": {},
   "source": [
    "<b> Creating timestamped grouped bar chart"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "ce321f43-2134-4da3-b124-b7307de6ee01",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_bar.index.name = 'Category'\n",
    "df_bar = df_bar.reset_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "1f5dcc4d-e5f6-4e09-bc18-ca5dddee2fc0",
   "metadata": {},
   "outputs": [],
   "source": [
    "melted = df_bar.melt(id_vars='Category', var_name='Metric', value_name='Value')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "732783c1-4ca9-42b2-8493-8f3c4ea48cf9",
   "metadata": {},
   "outputs": [],
   "source": [
    "melted.loc[melted['Metric'] == 'Total_Reviews', 'Value'] /= 1000000"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e512fa38-796e-4cc1-bcd9-247c45c5720b",
   "metadata": {},
   "source": [
    "<b>Plotting graph:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "4d52a59a-7801-45e2-bad2-1df9693d4a2d",
   "metadata": {},
   "outputs": [],
   "source": [
    "ist = pytz.timezone('Asia/Kolkata')\n",
    "now_ist = datetime.datetime.now(ist)\n",
    "\n",
    "if 15 <= now_ist.hour < 17:\n",
    "        # Figure 12\n",
    "        st.subheader(\"Average rating vs No. of reviews\")\n",
    "    \n",
    "        fig12 = px.bar(\n",
    "            melted,\n",
    "            x='Category',\n",
    "            y='Value',\n",
    "            color='Metric',\n",
    "            barmode='group',\n",
    "            title='Avg Rating vs Total Reviews(in Millions) (Top Categories - Jan)',\n",
    "            color_discrete_sequence=px.colors.qualitative.Vivid,\n",
    "            width=400,\n",
    "            height=300\n",
    "        )\n",
    "        fig12.update_layout(\n",
    "            plot_bgcolor='black',\n",
    "            paper_bgcolor='black',\n",
    "            font_color='white',\n",
    "            title_font={'size': 16},\n",
    "            xaxis=dict(title_font={'size': 12}),\n",
    "            yaxis=dict(title_font={'size': 12}),\n",
    "            margin=dict(l=10, r=10, t=30, b=10)\n",
    "        )\n",
    "\n",
    "        st.plotly_chart(fig12, use_container_width=True)\n",
    "\n",
    "else:\n",
    "    st.warning(\"⏰ This dashboard is only accessible from 3 PM to 5 PM IST. Please come back later.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "96c99dd8-873e-4ea3-9bbd-f8094d253644",
   "metadata": {},
   "source": [
    "### Task-3: Plot a time series line chart to show the trend of total installs over time, segmented by app category. Highlight periods of significant growth by shading the areas under the curve where the increase in installs exceeds 20% month-over-month and app name should not starts with x, y ,z and app category should start with letter \" E \" or \" C \" or \" B \" and We have to translate the Beauty category in Hindi and Business category in Tamil and Dating category in German while showing it on Graph. reviews should be more than 500 the app name should not contain letter \"S\" as well as this graph should work only between 6 PM IST to 9 PM IST apart from that time we should not show this graph in dashboard itself"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "04e59345-0ca6-4528-93d9-06e8d92978c0",
   "metadata": {},
   "outputs": [],
   "source": [
    "df3=apps_df.copy()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e24f1ca3-1dba-4b31-b38b-196bf9044c32",
   "metadata": {},
   "source": [
    "<b> Creating filtered data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "2aff142f-cb18-410c-94eb-6553d6d0bde0",
   "metadata": {},
   "outputs": [],
   "source": [
    "df3 = df3[\n",
    "    (~df3['App'].str[0].isin(['X', 'Y', 'Z'])) &  \n",
    "    (df3['Category'].str[0].isin(['E', 'C', 'B'])) &  \n",
    "    (df3['Reviews'] > 500) & \n",
    "    (~df3['App'].str.contains('S', case=False))\n",
    "]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b3356490-68fc-46f5-ac24-d79eb606e076",
   "metadata": {},
   "source": [
    "<b>Translating app names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "9a0f01f4-c1be-47b2-8f7e-208b5c404f58",
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM\n",
    "\n",
    "model_name = \"facebook/nllb-200-distilled-600M\"\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)\n",
    "model = AutoModelForSeq2SeqLM.from_pretrained(model_name)\n",
    "\n",
    "lang_code_map = {\n",
    "    'BEAUTY': 'hin_Deva',    # Hindi\n",
    "    'BUSINESS': 'tam_Taml',  # Tamil\n",
    "    'DATING': 'deu_Latn',    # German\n",
    "}\n",
    "\n",
    "def translate_category_nllb(category):\n",
    "    if category not in lang_code_map:\n",
    "        return category\n",
    "\n",
    "    tgt_lang = lang_code_map[category]\n",
    "    src_lang = 'eng_Latn'\n",
    "    \n",
    "    tokenizer.src_lang = src_lang\n",
    "    encoded = tokenizer(category, return_tensors=\"pt\")\n",
    "    generated_tokens = model.generate(\n",
    "        **encoded,\n",
    "        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang)\n",
    "    )\n",
    "    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c45289ca-d2db-446e-98ac-c642a5b26091",
   "metadata": {},
   "source": [
    "Dating category transformed into german is not present in current dataframe as it starts with \"D\" and not \"E,C or B\". So this condition is not included."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "99a2b517-74c7-4066-81e8-9b4ca39c477d",
   "metadata": {},
   "outputs": [],
   "source": [
    "df3['Translated_Category'] = df3['Category'].apply(translate_category_nllb)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "fdd8a6e8-c3c8-4a06-8e81-56ed9b098be4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Souhar\\AppData\\Local\\Temp\\ipykernel_44896\\1985472471.py:4: FutureWarning:\n",
      "\n",
      "'M' is deprecated and will be removed in a future version, please use 'ME' instead.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "df3 = df3.sort_values('Last Updated')\n",
    "\n",
    "grouped = df3.groupby(\n",
    "        ['Translated_Category', pd.Grouper(key='Last Updated', freq='M')]\n",
    "    )['Installs'].sum().reset_index()\n",
    "\n",
    "grouped['pct_change'] = grouped.groupby('Translated_Category')['Installs'].pct_change()\n",
    "grouped['high_growth'] = grouped['pct_change'] > 0.2\n",
    "grouped['month'] = grouped['Last Updated'].dt.strftime('%Y-%m')\n",
    "\n",
    "melted = grouped.rename(columns={\n",
    "    'Translated_Category': 'Category',\n",
    "    'Installs': 'Value'\n",
    "    })\n",
    "\n",
    "melted['Metric'] = melted['high_growth'].apply(lambda x: 'High Growth' if x else 'Normal Growth')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fddcb68e-d28d-4e24-8fad-9c91025bb8a6",
   "metadata": {},
   "source": [
    "<b>Plotting graph:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "b1066ac3-1c76-40ca-8d97-6266e27bb88f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import plotly.graph_objects as go\n",
    "\n",
    "ist = pytz.timezone('Asia/Kolkata')\n",
    "now_ist = datetime.datetime.now(ist)\n",
    "\n",
    "if 18 <= now_ist.hour < 21:\n",
    "    \n",
    "    # Figure 13\n",
    "    st.subheader(\"Monthly installs per category\")\n",
    "    \n",
    "    fig13 = px.line(\n",
    "        melted,\n",
    "        x='month',\n",
    "        y='Value',\n",
    "        color='Category',\n",
    "        line_group='Category',\n",
    "        title='Monthly Installs per Category (Translated Names)',\n",
    "        color_discrete_sequence=px.colors.qualitative.Set3,\n",
    "        width=1000,\n",
    "        height=500,\n",
    "        hover_data=['Metric']\n",
    "    )\n",
    "\n",
    "    color_map = {}\n",
    "    for trace in fig13.data:\n",
    "        category = trace.name\n",
    "        color = trace.line.color\n",
    "        color_map[category] = color  # Store the color used by each line\n",
    "\n",
    "    high_growth_data = melted[melted['Metric'] == 'High Growth']\n",
    "    categories = high_growth_data['Category'].unique()\n",
    "\n",
    "    for category in categories:\n",
    "        category_data = high_growth_data[high_growth_data['Category'] == category]\n",
    "        base_color = color_map.get(category, 'rgba(255,255,255,1)')\n",
    "\n",
    "        if base_color.startswith('#'):\n",
    "            hex_color = base_color.lstrip('#')\n",
    "            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))\n",
    "            rgba_color = f'rgba({r}, {g}, {b}, 0.3)'\n",
    "        else:\n",
    "            rgba_color = base_color.replace('1)', '0.3)') if 'rgba' in base_color else base_color\n",
    "\n",
    "        fig13.add_trace(go.Scatter(\n",
    "            x=category_data['month'],\n",
    "            y=category_data['Value'],\n",
    "            mode='lines',\n",
    "            name=f'{category} - High Growth',\n",
    "            fill='tozeroy',\n",
    "            line=dict(color=rgba_color),\n",
    "            showlegend=False,\n",
    "            hoverinfo='skip'\n",
    "        ))\n",
    "\n",
    "    fig13.update_layout(\n",
    "        plot_bgcolor='black',\n",
    "        paper_bgcolor='black',\n",
    "        font_color='white',\n",
    "        title_font={'size': 16},\n",
    "        xaxis=dict(title='Month', title_font={'size': 12}),\n",
    "        yaxis=dict(title='Total Installs', title_font={'size': 12}),\n",
    "        margin=dict(l=10, r=10, t=30, b=10)\n",
    "    )\n",
    "\n",
    "    st.plotly_chart(fig13, use_container_width=True)\n",
    "else:\n",
    "    st.warning(\"⏰ This dashboard is only accessible from 6 PM to 9 PM IST. Please come back later.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
