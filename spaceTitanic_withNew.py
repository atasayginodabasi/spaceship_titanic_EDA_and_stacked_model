from xgboost import XGBClassifier
import warnings
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings(action='ignore', category=UserWarning)

train = pd.read_csv('C:/Users/ata-d/OneDrive/Masaüstü/ML/Datasets/spaceTitanic_train.csv')
test = pd.read_csv(r'C:/Users/ata-d/OneDrive/Masaüstü/ML/Datasets/spaceTitanic_test.csv')

# ----------------------------------------------------------------------------------------------------------------------

# Count of the missing values of the dataset
count_NaN = train.isna().sum()

# ----------------------------------------------------------------------------------------------------------------------

# Filling missing Names with Unknown, I've found no usage area for them for now
train.Name = train.Name.fillna('Unknown')

# Dropping the rows those are both missing for HomePlanet and Cabin, I will try to fill the missing parts using either
# Cabin or HomePlanet
train.dropna(subset=['HomePlanet', 'Cabin'], how='all', inplace=True)

# ----------------------------------------------------------------------------------------------------------------------

# Creating three new columns based on Cabin Feature
train['Deck'] = train.Cabin.str.split('/').str[0]
train['Num'] = train.Cabin.str.split('/').str[1]
train['Side'] = train.Cabin.str.split('/').str[2]
train.drop(['Cabin'], axis=1, inplace=True)

# ----------------------------------------------------------------------------------------------------------------------

# Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with
train['GroupNumber'] = train.PassengerId.str.split('_').str[0]

# Let's check for if same GroupNumber's are departed from same HomePlanet,
# so I would fill the missing HomePlanet values using GroupNumber
Group_HomePlanet = train.groupby(['GroupNumber', 'HomePlanet'])['HomePlanet'].size().unstack().fillna(0).reset_index() \
    .sort_values('GroupNumber')

# Subtracted 1 because, GroupNumber is also a nonzero and I don't want to count it
Group_HomePlanet['UniqueHomePlanets'] = np.count_nonzero(Group_HomePlanet, axis=1) - 1

# Each Group are departed from the same HomePlanet. So, I would fill the nan HomePlanet values using GroupNumbers
print('Number of Unique HomePlanet(s) of each Group:', Group_HomePlanet['UniqueHomePlanets'].unique())

# Creating a dictionary for each GroupNumber: HomePlanet, so I will replace the missing these values using GroupNumber
Group_HomePlanet_dict = train[['GroupNumber', 'HomePlanet']].drop_duplicates().dropna().set_index('GroupNumber')
Group_HomePlanet_dict = Group_HomePlanet_dict.to_dict()['HomePlanet']
train.HomePlanet = train.GroupNumber.map(Group_HomePlanet_dict)

# There are still missing values on HomePlanet, so I will use other Columns for filling the missing values.
train.isna().sum()

# ------------------------------------

# Checking Number of Passengers in each Deck. Some Decks include Passengers from only one HomePlanet, so
# I will fill missing values like that
train.groupby(['HomePlanet', 'Deck'])['Deck'].size().unstack().fillna(0)

# Filling missing HomePlanet value using Decks. A, B, C and T Decks only include Europa Passengers
train.loc[(train.HomePlanet.isna() & (train.Deck.isin(['A', 'B', 'C', 'T']))), 'HomePlanet'] = \
    train.loc[(train.HomePlanet.isna() & (train.Deck.isin(['A', 'B', 'C', 'T']))), 'HomePlanet'].fillna('Europa')

# Most of the E, F and G Deck Passengers are from Earth, so I will replace missing Planet values using these Decks
train.loc[(train.HomePlanet.isna() & (train.Deck.isin(['E', 'F', 'G']))), 'HomePlanet'] = \
    train.loc[(train.HomePlanet.isna() & (train.Deck.isin(['E', 'F', 'G']))), 'HomePlanet'].fillna('Earth')

# Most of the D Deck Passengers are from Mars, so I will replace missing Planet values using this Deck
train.loc[(train.HomePlanet.isna() & (train.Deck.isin(['D']))), 'HomePlanet'] = \
    train.loc[(train.HomePlanet.isna() & (train.Deck.isin(['D']))), 'HomePlanet'].fillna('Mars')

# There are no missing HomePlanet values now
train.isna().sum()

# ------------------------------------

# Checking for HomePlanet and Deck Relationship for filling missing Deck values
print(train.groupby(['HomePlanet', 'Deck'])['Deck'].count().unstack().fillna(0))

# Filling missing Deck values with the most frequent Deck category for each HomePlanet
train.loc[(train.Deck.isna() & (train.HomePlanet == 'Earth')), 'Deck'] = \
    train.loc[(train.Deck.isna() & (train.HomePlanet == 'Earth')), 'Deck'].fillna('G')

train.loc[(train.Deck.isna() & (train.HomePlanet == 'Europa')), 'Deck'] = \
    train.loc[(train.Deck.isna() & (train.HomePlanet == 'Europa')), 'Deck'].fillna('B')

train.loc[(train.Deck.isna() & (train.HomePlanet == 'Mars')), 'Deck'] = \
    train.loc[(train.Deck.isna() & (train.HomePlanet == 'Mars')), 'Deck'].fillna('F')

# There are no missing Deck values now
train.isna().sum()

# ------------------------------------

# Let's check for if same GroupNumber's are located in same Side of the ship
Group_Side = train.groupby(['GroupNumber', 'Side'])['Side'].size().unstack().fillna(0).reset_index() \
    .sort_values('GroupNumber')

# Subtracted 1 because, GroupNumber is also a nonzero and I don't want to count it
Group_Side['UniqueSides'] = np.count_nonzero(Group_Side, axis=1) - 1

# Each Group are located in the same Side of the Ship because their passengers are located only one
# unique Side of the Ship. So, I would fill the nan Side values using GroupNumbers
print('Number of Unique Side Location(s) of each Group:', Group_Side['UniqueSides'].unique())

# Creating a dictionary for each GroupNumber: Side, so I will replace the missing Side values using GroupNumber
Group_Side_dict = train[['GroupNumber', 'Side']].drop_duplicates().dropna().set_index('GroupNumber')
Group_Side_dict = Group_Side_dict.to_dict()['Side']
train.Side = train.GroupNumber.map(Group_Side_dict)

# Dropping missing Side and Num values (there was 193 missing values, it's 95 now)
train.dropna(subset=['Side', 'Num'], axis=0, inplace=True)

# There are no missing Side and Num values now
train.isna().sum()

# ------------------------------------

Group_Destination = train.groupby(['GroupNumber', 'Destination'])['Destination'].size().unstack().fillna(0) \
    .reset_index().sort_values('GroupNumber')
Group_Destination['UniqueDestinations'] = np.count_nonzero(Group_Destination, axis=1) - 1

# 88.44% of the Groups are going to same Destination, and 11.6% of the Groups have at least one Passenger that going
# different Destination from other Group members.
print((Group_Destination['UniqueDestinations'].value_counts() /
       Group_Destination['UniqueDestinations'].value_counts().sum()) * 100)

# I can fill the missing Destination data points of only Groups have 1 UniqueDestinations, if I do it for all (2, 3),
# I will be losing the Destination differences in-Group. Of course, I don't want to lose that information.

# So, let's make our shot for filling the missing Destination data points of only Groups have 1 UniqueDestinations,
# and I hope we would recover some of the missing 178 Destination data.
OneUniqueDestination_list = Group_Destination[Group_Destination.UniqueDestinations == 1].GroupNumber

# Filtering the train data for only Groups that have One Unique Destination, and then I will create a dict for fill nan
OneUniqueDestination = train[train.GroupNumber.isin(OneUniqueDestination_list)][['GroupNumber', 'Destination']] \
    .dropna().drop_duplicates().set_index('GroupNumber')
OneUniqueDestination_dict = OneUniqueDestination.to_dict()['Destination']

# Filling missing Dest. rows with this condition: missing Destination column AND GroupNumber is in OneUniqueDestination.
# Hence, I replaced the missing Destination values using their GroupNumber's for only Groups that have only one
# unique Destination, in order to preserve the in-Group Destination features.
train.loc[(train.Destination.isna() & (train.GroupNumber.isin(OneUniqueDestination.index))), 'Destination'] = \
    train.loc[(train.Destination.isna() & (train.GroupNumber.isin(OneUniqueDestination.index)))].GroupNumber \
        .map(OneUniqueDestination_dict)

# I saved 50 Destination rows by application above. Unfortunately, I will drop others
train.isna().sum()
train.dropna(subset=['Destination'], axis=0, inplace=True)

# ------------------------------------

# Creating a Column based on people's Journey Route: HomePlanet + Destination
train['JourneyRoute'] = train.HomePlanet + ' to ' + train.Destination

# ------------------------------------

# Let's check if there is a relation between Journey Route and VIP status with the Prices paid
# Average Room Service Price for each JourneyRouse and VIP status
AvgRoom = np.ceil(train.groupby(['JourneyRoute', 'VIP'])['RoomService'].mean().unstack().replace(0, np.nan))

# Average Room Services vs. Journey Routes
'''''''''
fig = go.Figure()
fig.add_trace(go.Bar(
    y=AvgRoom[AvgRoom.columns[0]],
    x=AvgRoom.index,
    text=AvgRoom[AvgRoom.columns[0]].round(2),
    textposition='outside',
    name="non-VIP"))

fig.add_trace(go.Bar(
    y=AvgRoom[AvgRoom.columns[1]],
    x=AvgRoom.index,
    text=AvgRoom[AvgRoom.columns[1]],
    textposition='outside',
    name=f"VIP"))

fig.update_layout(
    yaxis=dict(title_text="Average Room Service", titlefont=dict(size=15)),
    xaxis=dict(title_text="Journey Routes", titlefont=dict(size=15)),
    title={'text': f"Average Room Services vs. Journey Routes",
           'x': 0.5})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)), texttemplate='%{text:.4s}')
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_layout(xaxis={'categoryorder': 'total descending'}, barmode='stack')
fig.show()
'''''''''

# Filling missing RoomService values by their Journey Route and VIP's status mean (LONG VERSION)
for i in range(AvgRoom.index.shape[0]):
    for j in range(2):
        train.loc[(train.RoomService.isna() & (train.JourneyRoute == AvgRoom.index[i]) & (train.VIP == j)),
                  'RoomService'] = train.loc[(train.RoomService.isna() & (train.JourneyRoute == AvgRoom.index[i])
                                              & (train.VIP == j)), 'RoomService'].fillna(AvgRoom[j][AvgRoom.index[i]])

# Short Version (BE CAREFULL, eğer groupby'ladığın columnlarda nan değerler varsa doğruluğunu kontrol et)
# train.RoomService = train.groupby(['JourneyRoute', 'VIP']).RoomService.apply(lambda k: np.ceil(k.fillna(k.mean())))

# ------

# Average FoodCourt Price for each JourneyRouse and VIP status
AvgFood = np.ceil(train.groupby(['JourneyRoute', 'VIP'])['FoodCourt'].mean().unstack().replace(0, np.nan))

# Average Food Court vs. Journey Routes
'''''''''
fig = go.Figure()
fig.add_trace(go.Bar(
    y=AvgFood[AvgFood.columns[0]],
    x=AvgFood.index,
    text=AvgFood[AvgFood.columns[0]].round(2),
    textposition='outside',
    name="non-VIP"))

fig.add_trace(go.Bar(
    y=AvgFood[AvgFood.columns[1]],
    x=AvgFood.index,
    text=AvgFood[AvgFood.columns[1]],
    textposition='outside',
    name=f"VIP"))

fig.update_layout(
    yaxis=dict(title_text="Average Food Court", titlefont=dict(size=15)),
    xaxis=dict(title_text="Journey Routes", titlefont=dict(size=15)),
    title={'text': f"Average Food Court vs. Journey Routes",
           'x': 0.5})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)), texttemplate='%{text:.4s}')
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_layout(xaxis={'categoryorder': 'total descending'}, barmode='stack')
fig.show()
'''''''''

# Filling missing FoodCourt values by their Journey Route and VIP's status mean (LONG VERSION)
for i in range(AvgFood.index.shape[0]):
    for j in range(2):
        train.loc[(train.FoodCourt.isna() & (train.JourneyRoute == AvgFood.index[i]) & (train.VIP == j)),
                  'FoodCourt'] = train.loc[(train.FoodCourt.isna() & (train.JourneyRoute == AvgFood.index[i])
                                            & (train.VIP == j)), 'FoodCourt'].fillna(AvgFood[j][AvgFood.index[i]])

# Short Version
# train.FoodCourt = train.groupby(['JourneyRoute', 'VIP']).FoodCourt.apply(lambda k: np.ceil(k.fillna(k.mean())))

# ------

# Average ShoppingMall Price for each JourneyRouse and VIP status
AvgShopping = np.ceil(train.groupby(['JourneyRoute', 'VIP'])['ShoppingMall'].mean().unstack().replace(0, np.nan))

# Average ShoppingMall vs. Journey Routes
'''''''''
fig = go.Figure()
fig.add_trace(go.Bar(
    y=AvgShopping[AvgShopping.columns[0]],
    x=AvgShopping.index,
    text=AvgShopping[AvgShopping.columns[0]].round(2),
    textposition='outside',
    name="non-VIP"))

fig.add_trace(go.Bar(
    y=AvgShopping[AvgShopping.columns[1]],
    x=AvgShopping.index,
    text=AvgShopping[AvgShopping.columns[1]],
    textposition='outside',
    name=f"VIP"))

fig.update_layout(
    yaxis=dict(title_text="Average Shopping", titlefont=dict(size=15)),
    xaxis=dict(title_text="Journey Routes", titlefont=dict(size=15)),
    title={'text': f"Average Shopping vs. Journey Routes",
           'x': 0.5})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)), texttemplate='%{text:.4s}')
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_layout(xaxis={'categoryorder': 'total descending'}, barmode='stack')
fig.show()
'''''''''

# Filling missing ShoppingMall values by their Journey Route and VIP's status mean (LONG VERSION)
for i in range(AvgShopping.index.shape[0]):
    for j in range(2):
        train.loc[(train.ShoppingMall.isna() & (train.JourneyRoute == AvgShopping.index[i]) & (train.VIP == j)),
                  'ShoppingMall'] = train.loc[(train.ShoppingMall.isna() & (train.JourneyRoute == AvgShopping.index[i])
                                               & (train.VIP == j)), 'ShoppingMall'] \
            .fillna(AvgShopping[j][AvgShopping.index[i]])

# Short Version
# train.ShoppingMall = train.groupby(['JourneyRoute', 'VIP']).ShoppingMall.apply(lambda k: np.ceil(k.fillna(k.mean())))

# ------

# Average Spa Price for each JourneyRouse and VIP status
AvgSpa = np.ceil(train.groupby(['JourneyRoute', 'VIP'])['Spa'].mean().unstack().replace(0, np.nan))

# Average Spa vs. Journey Routes
'''''''''
fig = go.Figure()
fig.add_trace(go.Bar(
    y=AvgSpa[AvgSpa.columns[0]],
    x=AvgSpa.index,
    text=AvgSpa[AvgSpa.columns[0]].round(2),
    textposition='outside',
    name="non-VIP"))

fig.add_trace(go.Bar(
    y=AvgSpa[AvgSpa.columns[1]],
    x=AvgSpa.index,
    text=AvgSpa[AvgSpa.columns[1]],
    textposition='outside',
    name=f"VIP"))

fig.update_layout(
    yaxis=dict(title_text="Average Spa", titlefont=dict(size=15)),
    xaxis=dict(title_text="Journey Routes", titlefont=dict(size=15)),
    title={'text': f"Average Spa vs. Journey Routes",
           'x': 0.5})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)), texttemplate='%{text:.4s}')
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_layout(xaxis={'categoryorder': 'total descending'}, barmode='stack')
fig.show()
'''''''''

# Filling missing Spa values by their Journey Route and VIP's status mean (LONG VERSION)
for i in range(AvgSpa.index.shape[0]):
    for j in range(2):
        train.loc[(train.Spa.isna() & (train.JourneyRoute == AvgSpa.index[i]) & (train.VIP == j)), 'Spa'] = \
            train.loc[(train.Spa.isna() & (train.JourneyRoute == AvgSpa.index[i]) & (train.VIP == j)), 'Spa'] \
                .fillna(AvgSpa[j][AvgSpa.index[i]])

# Short Version
# train.Spa = train.groupby(['JourneyRoute', 'VIP']).Spa.apply(lambda k: np.ceil(k.fillna(k.mean())))

# ------

# Average VRDeck Price for each JourneyRouse and VIP status
AvgVRDeck = np.ceil(train.groupby(['JourneyRoute', 'VIP'])['VRDeck'].mean().unstack().replace(0, np.nan))

# Average VRDeck vs. Journey Routes
'''''''''
fig = go.Figure()
fig.add_trace(go.Bar(
    y=AvgVRDeck[AvgVRDeck.columns[0]],
    x=AvgVRDeck.index,
    text=AvgVRDeck[AvgVRDeck.columns[0]].round(2),
    textposition='outside',
    name="non-VIP"))

fig.add_trace(go.Bar(
    y=AvgVRDeck[AvgVRDeck.columns[1]],
    x=AvgVRDeck.index,
    text=AvgVRDeck[AvgVRDeck.columns[1]],
    textposition='outside',
    name=f"VIP"))

fig.update_layout(
    yaxis=dict(title_text="Average VR Deck", titlefont=dict(size=15)),
    xaxis=dict(title_text="Journey Routes", titlefont=dict(size=15)),
    title={'text': f"Average VR Deck vs. Journey Routes",
           'x': 0.5})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)), texttemplate='%{text:.4s}')
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_layout(xaxis={'categoryorder': 'total descending'}, barmode='stack')
fig.show()
'''''''''

# Filling missing VRDeck values by their Journey Route and VIP's status mean (LONG VERSION)
for i in range(AvgVRDeck.index.shape[0]):
    for j in range(2):
        train.loc[(train.VRDeck.isna() & (train.JourneyRoute == AvgVRDeck.index[i]) & (train.VIP == j)), 'VRDeck'] = \
            train.loc[(train.VRDeck.isna() & (train.JourneyRoute == AvgVRDeck.index[i]) & (train.VIP == j)), 'VRDeck'] \
                .fillna(AvgVRDeck[j][AvgVRDeck.index[i]])

# Short Version
# train.VRDeck = train.groupby(['JourneyRoute', 'VIP']).VRDeck.apply(lambda k: np.ceil(k.fillna(k.mean())))

# ------------------------------------

# Creating a Total Expenses column for summation of all Expenses
train['TotalExpenses'] = train.RoomService + train.FoodCourt + train.ShoppingMall + train.Spa + train.VRDeck

# Remaining missing rows after all these operations
train.isna().sum()

# ------------------------------------

# Let's check missing Total Expenses values, I hope we'd find some relation for filling missing further values here
# There're some CryoSleep positive vals there, they usually don't spend on any categories , so we'd fill them with zero.
# At the end, I saved 6 rows using this code block below
# train[(train.TotalExpenses.isna()) & (train.CryoSleep == 1)]

train.loc[(train.TotalExpenses.isna() & (train.CryoSleep == 1)), ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa',
                                                                  'VRDeck']] = \
    train.loc[(train.TotalExpenses.isna() & (train.CryoSleep == 1)), ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa',
                                                                      'VRDeck']].fillna(0)

# Update
train['TotalExpenses'] = train.RoomService + train.FoodCourt + train.ShoppingMall + train.Spa + train.VRDeck

train.dropna(subset=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=0, inplace=True)

# ------------------------------------

# Assigning missing Age values by thier HomePlanet's average Age value
train.Age = train.groupby('HomePlanet').Age.apply(lambda k: np.ceil(k.fillna(k.mean())))

# ------------------------------------

train.dropna(subset=['CryoSleep', 'VIP'], axis=0, inplace=True)

# ----------------------------------------------------------------------------------------------------------------------

# Data Visualization
data_vizz = train.copy()
data_vizz.VIP = np.where(data_vizz.VIP == 1, 'VIP', 'Not-VIP')
data_vizz.Transported = np.where(data_vizz.Transported == 1, 'Transported', 'Not-Transported')
data_vizz.CryoSleep = np.where(data_vizz.CryoSleep == 1, 'Cryo Sleeping', 'Not-Cryo Sleeping')

# Journey Routes of the Passengers
'''''''''
fig = px.bar(data_frame=train, x=train['JourneyRoute'].unique(), y=train['JourneyRoute'].value_counts(),
             color=train['JourneyRoute'].unique(), text=train['JourneyRoute'].value_counts(),
             labels={
                 "x": "Journey Route",
                 "y": "Number of Passengers"})
fig.update_layout(title_text='Journey Routes of the Passengers',
                  title_x=0.5, title_font=dict(size=20))
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)), textposition='outside')
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.show()
'''''''''

# Journey Route and VIP for Transportation
'''''''''
fig = px.sunburst(data_frame=data_vizz,
                  path=["JourneyRoute", "VIP", "Transported"],
                  maxdepth=-1,
                  branchvalues='total',
                  hover_data={'Transported': False},
                  title='Journey Route and VIP for Transportation', template='ggplot2')

fig.update_traces(textinfo='label+percent parent')
fig.update_layout(font=dict(size=14))
fig.show()
'''''''''

# Journey Routes by Transportation Success Ratio
'''''''''
JR_T = data_vizz.groupby(['JourneyRoute', 'Transported']).size().unstack().apply(lambda l: l/l.sum(), axis=1)
fig = px.bar(JR_T, y=JR_T.index, x=['Transported', 'Not-Transported'],
             barmode='stack', text='value',
             labels={"value": "Ratio of the Transportation Success [%]"})
fig.update_layout(title_text='Journey Routes by Transportation Success Ratio',
                  title_x=0.5, title_font=dict(size=20))
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_layout(xaxis_tickformat='%')
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.update_traces(texttemplate='%{text:.1%f}', textposition='inside')
fig.show()
'''''''''

# Ratio of Cryo Sleep Passengers
'''''''''
fig = px.pie(train, names='CryoSleep')
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.update_layout(title_text='Ratio of the Cryo Sleep Passengers', title_x=0.5)
fig.update_layout(font=dict(size=14))
fig.show()
'''''''''

# Cryo Sleep by Transportation Success Ratio
'''''''''
JR_T = data_vizz.groupby(['CryoSleep', 'Transported']).size().unstack().apply(lambda l: l/l.sum(), axis=1)
fig = px.bar(JR_T, y=JR_T.index, x=['Transported', 'Not-Transported'],
             barmode='stack', text='value',
             labels={"value": "Ratio of the Transportation Success [%]"})
fig.update_layout(title_text='CryoSleep by Transportation Success Ratio',
                  title_x=0.5, title_font=dict(size=20))
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_layout(xaxis_tickformat='%')
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.update_traces(texttemplate='%{text:.1%f}', textposition='inside')
fig.show()
'''''''''

# Deck by Transportation Success Ratio
'''''''''
JR_T = data_vizz.groupby(['Deck', 'Transported']).size().unstack().apply(lambda l: l/l.sum(), axis=1)
fig = px.bar(JR_T, y=JR_T.index, x=['Transported', 'Not-Transported'],
             barmode='stack', text='value',
             labels={"value": "Ratio of the Transportation Success [%]"})
fig.update_layout(title_text='Deck by Transportation Success Ratio',
                  title_x=0.5, title_font=dict(size=20))
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_layout(xaxis_tickformat='%')
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.update_traces(texttemplate='%{text:.1%f}', textposition='inside')
fig.show()
'''''''''

# Distribution of the Age by Transportation Success
'''''''''
fig = px.histogram(data_vizz, x="Age", color='Transported', opacity=0.8, marginal='box')
fig.update_layout(barmode='overlay', xaxis={'categoryorder': 'total descending'})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.update_layout(title_text='Distribution of Math Scores by Gender',
                  title_x=0.5, title_font=dict(size=20))
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.show()
'''''''''

# Distribution of the Age by VIP Status
'''''''''
fig = px.histogram(data_vizz, x="Age", color='VIP', opacity=0.8, marginal='box')
fig.update_layout(barmode='overlay', xaxis={'categoryorder': 'total descending'})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.update_layout(title_text='Distribution of Math Scores by Gender',
                  title_x=0.5, title_font=dict(size=20))
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.show()
'''''''''

# Distribution of the Age by Cryo Sleep
'''''''''
fig = px.histogram(data_vizz, x="Age", color='CryoSleep', opacity=0.7, marginal='box')
fig.update_layout(barmode='overlay', xaxis={'categoryorder': 'total descending'})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.update_layout(title_text='Distribution of Math Scores by Gender',
                  title_x=0.5, title_font=dict(size=20))
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.show()
'''''''''

# Average TotalExpenses vs. Journey Routes
'''''''''
AvgTotalExpenses = train.groupby(['JourneyRoute', 'VIP']).TotalExpenses.mean().unstack().replace(0, np.nan)
fig = go.Figure()
fig.add_trace(go.Bar(
    y=AvgTotalExpenses[AvgTotalExpenses.columns[0]],
    x=AvgTotalExpenses.index,
    text=AvgTotalExpenses[AvgTotalExpenses.columns[0]].round(2),
    textposition='outside',
    name="non-VIP"))

fig.add_trace(go.Bar(
    y=AvgTotalExpenses[AvgTotalExpenses.columns[1]],
    x=AvgTotalExpenses.index,
    text=AvgTotalExpenses[AvgTotalExpenses.columns[1]],
    textposition='outside',
    name=f"VIP"))

fig.update_layout(
    yaxis=dict(title_text="Average Total Expenses", titlefont=dict(size=15)),
    xaxis=dict(title_text="Journey Routes", titlefont=dict(size=15)),
    title={'text': f"Average Total Expenses vs. Journey Routes",
           'x': 0.5})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)), texttemplate='%{text:.4s}')
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_layout(xaxis={'categoryorder': 'total descending'}, barmode='stack')
fig.show()
'''''''''

# Outliers of the TotalExpense
'''''''''
fig = px.box(data_vizz, y=['TotalExpenses'], color='VIP',
             labels={'variable': 'VIP'})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.update_layout(title_text='Outliers of the TotalExpense',
                  title_x=0.5, title_font=dict(size=14))
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.show()
'''''''''

# Correlation Analysis of the Dataset
'''''''''
plt.figure(figsize=(15, 8))
heatmap = sns.heatmap(train.corr(), vmin=-1, vmax=1, annot=True, linewidths=1, linecolor='black', cmap="BuPu")
heatmap.set_title('Correlation Graph of the Training Dataset', fontdict={'fontsize': 14})
'''''''''

# ----------------------------------------------------------------------------------------------------------------------

train['RatioRoomService'] = (train['RoomService'] / train['TotalExpenses']).fillna(0)
train['RatioFoodCourt'] = (train['FoodCourt'] / train['TotalExpenses']).fillna(0)
train['RatioShoppingMall'] = (train['ShoppingMall'] / train['TotalExpenses']).fillna(0)
train['RatioSpa'] = (train['Spa'] / train['TotalExpenses']).fillna(0)
train['RatioVRDeck'] = (train['VRDeck'] / train['TotalExpenses']).fillna(0)

# Is_Single Feature. Returns Single if Passenger has no other Group Member
train['InGroupNumber'] = train.PassengerId.str.split('_').str[1]
x = train.groupby('GroupNumber').InGroupNumber.apply(lambda Z: 'Single' if Z.max() == '01' else 'Not-Single') \
    .reset_index()
x.columns = ['GroupNumber', 'Is_Single']
train = train.merge(x, on='GroupNumber')
train.drop('InGroupNumber', axis=1, inplace=True)

# GroupSize Feature. Returns Size of each Group
train['InGroupNumber'] = train.PassengerId.str.split('_').str[1]
x = train.groupby('GroupNumber').InGroupNumber.apply(lambda Z: Z.max()).reset_index()
x.columns = ['GroupNumber', 'GroupSize']
train = train.merge(x, on='GroupNumber')
train.drop('InGroupNumber', axis=1, inplace=True)
train.GroupSize = train.GroupSize.astype(int)

train.GroupSize = pd.cut(train.GroupSize, 3, labels=['Small', 'Medium', 'Large'])

train['Is_Spend'] = np.where(train.TotalExpenses > 0, 'Spent', 'Not-Spent')

train.Age = pd.qcut(train.Age, 5, labels=['A', 'B', 'C', 'D', 'E'])

train['RoomService'] = StandardScaler().fit_transform(np.array(train['RoomService']).reshape(-1, 1))
train['FoodCourt'] = StandardScaler().fit_transform(np.array(train['FoodCourt']).reshape(-1, 1))
train['ShoppingMall'] = StandardScaler().fit_transform(np.array(train['ShoppingMall']).reshape(-1, 1))
train['Spa'] = StandardScaler().fit_transform(np.array(train['Spa']).reshape(-1, 1))
train['VRDeck'] = StandardScaler().fit_transform(np.array(train['VRDeck']).reshape(-1, 1))
train['TotalExpenses'] = StandardScaler().fit_transform(np.array(train['TotalExpenses']).reshape(-1, 1))

# Train dataset encoding
train_encoded = train[['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService',
                       'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported', 'Deck', 'Side', 'TotalExpenses',
                       'Is_Spend', 'GroupSize',
                       'RatioRoomService', 'RatioFoodCourt', 'RatioShoppingMall', 'RatioSpa', 'RatioVRDeck'
                       ]]

train_encoded = pd.get_dummies(train_encoded)  # One Hot Encoder by Pandas

# Dividing Data into Features and Labels
y = train_encoded['Transported']  # y values are the values that I want to predict.
X = train_encoded.drop('Transported', axis=1)

# Train and Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=13)

# ----------------------------------------------------------------------------------------------------------------------

# Grid Search
'''''''''
param_grid = {
    'learning_rate': [0.1, 0.15, 0.2],
    'n_estimators': [100, 120, 130],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3],
    'max_depth': [3, 5, 7]
    }
rf = GradientBoostingClassifier()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, n_jobs=2, verbose=2)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_grid = grid_search.best_estimator_
best_score = grid_search.best_score_
'''''''''

# -------------
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.BatchNormalization(input_shape=[X_train.shape[1]]),
    layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True)

hist = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=100,
    callbacks=[early_stopping])

# Train and Validation Loss Graphs
plt.figure(figsize=(10, 5))
plt.plot(hist.history['loss'], label='Train Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Train and Validation Loss Graphs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

prediction_train = model.predict(X_train)
prediction_val = model.predict(X_val)


def errors(actual, prediction):  # Error functions for the next steps

    m = keras.metrics.BinaryAccuracy()

    m.update_state(actual, prediction)

    error = m.result().numpy()  # MAPE

    return (error*100).round(3)


print('Train Error: %', errors(y_train, prediction_train))
print('Validation Error: %', errors(y_val, prediction_val))

# ----------------------------------------------------------------------------------------------------------------------


# Cleaning Test Data
def cleaning_test(test):
    test = pd.read_csv(r'C:/Users/ata-d/OneDrive/Masaüstü/ML/Datasets/spaceTitanic_test.csv')

    # Filling missing Names with Unknown, I've found no usage area for them for now
    test.Name = test.Name.fillna('Unknown')

    # ---------------

    # Creating three new columns based on Cabin Feature
    test['Deck'] = test.Cabin.str.split('/').str[0]
    test['Side'] = test.Cabin.str.split('/').str[2]
    test.drop(['Cabin'], axis=1, inplace=True)

    # Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with
    test['GroupNumber'] = test.PassengerId.str.split('_').str[0]

    # ---------------

    # There are no common GroupNames for train and test dataset
    train.merge(test, on='GroupNumber')

    # ---------------

    Group_HomePlanet_test = test.groupby(['GroupNumber', 'HomePlanet'])['HomePlanet'].size().unstack().fillna(0) \
        .reset_index().sort_values('GroupNumber')

    # Subtracted 1 because, GroupNumber is also a nonzero and I don't want to count it
    Group_HomePlanet_test['UniqueHomePlanets'] = np.count_nonzero(Group_HomePlanet_test, axis=1) - 1

    # Each Group are departed from the same HomePlanet. So, I would fill the nan HomePlanet values using GroupNumbers
    # print('Number of Unique HomePlanet(s) of each Group:', Group_HomePlanet_test['UniqueHomePlanets'].unique())

    # Creating a dictionary for each GroupNumber: HomePlanet, so I will replace the missing these values using GroupNumber
    Group_HomePlanet_dict_test = test[['GroupNumber', 'HomePlanet']].drop_duplicates().dropna().set_index('GroupNumber')
    Group_HomePlanet_dict_test = Group_HomePlanet_dict_test.to_dict()['HomePlanet']
    test.HomePlanet = test.GroupNumber.map(Group_HomePlanet_dict_test)

    # ---------------

    # Checking Number of Passengers in each Deck. Some Decks include Passengers from only one HomePlanet, so
    # I will fill missing values like that
    test.groupby(['HomePlanet', 'Deck'])['Deck'].size().unstack().fillna(0)

    # Filling missing HomePlanet value using Decks. A, B, C and T Decks only include Europa Passengers
    test.loc[(test.HomePlanet.isna() & (test.Deck.isin(['A', 'B', 'C', 'T']))), 'HomePlanet'] = \
        test.loc[(test.HomePlanet.isna() & (test.Deck.isin(['A', 'B', 'C', 'T']))), 'HomePlanet'].fillna('Europa')

    # Most of the E, F and G Deck Passengers are from Earth, so I will replace missing Planet values using these Decks
    test.loc[(test.HomePlanet.isna() & (test.Deck.isin(['E', 'F', 'G']))), 'HomePlanet'] = \
        test.loc[(test.HomePlanet.isna() & (test.Deck.isin(['E', 'F', 'G']))), 'HomePlanet'].fillna('Earth')

    # Most of the D Deck Passengers are from Mars, so I will replace missing Planet values using this Deck
    test.loc[(test.HomePlanet.isna() & (test.Deck.isin(['D']))), 'HomePlanet'] = \
        test.loc[(test.HomePlanet.isna() & (test.Deck.isin(['D']))), 'HomePlanet'].fillna('Mars')

    test.HomePlanet.fillna('Europa', inplace=True)

    # ---------------

    # Checking for HomePlanet and Deck Relationship for filling missing Deck values
    test.groupby(['HomePlanet', 'Deck'])['Deck'].count().unstack().fillna(0)

    # Filling missing Deck values with the most frequent Deck category for each HomePlanet
    test.loc[(test.Deck.isna() & (test.HomePlanet == 'Earth')), 'Deck'] = \
        test.loc[(test.Deck.isna() & (test.HomePlanet == 'Earth')), 'Deck'].fillna('G')

    test.loc[(test.Deck.isna() & (test.HomePlanet == 'Europa')), 'Deck'] = \
        test.loc[(test.Deck.isna() & (test.HomePlanet == 'Europa')), 'Deck'].fillna('B')

    test.loc[(test.Deck.isna() & (test.HomePlanet == 'Mars')), 'Deck'] = \
        test.loc[(test.Deck.isna() & (test.HomePlanet == 'Mars')), 'Deck'].fillna('F')

    # ---------------

    # Assigning missing Age values by thier HomePlanet's average Age value
    test.Age = test.groupby('HomePlanet').Age.apply(lambda k: np.ceil(k.fillna(k.mean())))

    # ---------------

    # Let's check for if same GroupNumber's are located in same Side of the ship
    Group_Side_test = test.groupby(['GroupNumber', 'Side'])['Side'].size().unstack().fillna(0).reset_index() \
        .sort_values('GroupNumber')

    # Subtracted 1 because, GroupNumber is also a nonzero and I don't want to count it
    Group_Side_test['UniqueSides'] = np.count_nonzero(Group_Side_test, axis=1) - 1

    # Each Group are located in the same Side of the Ship because their passengers are located only one
    # unique Side of the Ship. So, I would fill the nan Side values using GroupNumbers
    # print('Number of Unique Side Location(s) of each Group:', Group_Side_test['UniqueSides'].unique())

    # Creating a dictionary for each GroupNumber: Side, so I will replace the missing Side values using GroupNumber
    Group_Side_dict_test = test[['GroupNumber', 'Side']].drop_duplicates().dropna().set_index('GroupNumber')
    Group_Side_dict_test = Group_Side_dict_test.to_dict()['Side']
    test.Side = test.GroupNumber.map(Group_Side_dict_test)

    test.Side.fillna('P', inplace=True)

    # ---------------

    Group_Destination_test = test.groupby(['GroupNumber', 'Destination'])['Destination'].size().unstack().fillna(0) \
        .reset_index().sort_values('GroupNumber')
    Group_Destination_test['UniqueDestinations'] = np.count_nonzero(Group_Destination_test, axis=1) - 1

    OneUniqueDestination_list_test = Group_Destination_test[Group_Destination_test.UniqueDestinations == 1].GroupNumber

    # Filtering the train data for only Groups that have One Unique Destination, and then I will create a dict for fill nan
    OneUniqueDestination_list_test = test[test.GroupNumber.isin(OneUniqueDestination_list_test)][['GroupNumber',
                                                                                                  'Destination']].dropna() \
        .drop_duplicates().set_index('GroupNumber')
    OneUniqueDestination_dict_test = OneUniqueDestination_list_test.to_dict()['Destination']

    test.loc[(test.Destination.isna() & (test.GroupNumber.isin(OneUniqueDestination_list_test.index))), 'Destination'] = \
        test.loc[(test.Destination.isna() & (test.GroupNumber.isin(OneUniqueDestination_list_test.index)))].GroupNumber \
            .map(OneUniqueDestination_dict_test)

    test.loc[(test.Destination.isna() & (test.HomePlanet == 'Earth')), 'Destination'] = \
        test.loc[(test.Destination.isna() & (test.HomePlanet == 'Earth')), 'Destination'].fillna('TRAPPIST-1e')

    test.loc[(test.Destination.isna() & (test.HomePlanet == 'Europa')), 'Destination'] = \
        test.loc[(test.Destination.isna() & (test.HomePlanet == 'Europa')), 'Destination'].fillna('TRAPPIST-1e')

    test.loc[(test.Destination.isna() & (test.HomePlanet == 'Mars')), 'Destination'] = \
        test.loc[(test.Destination.isna() & (test.HomePlanet == 'Mars')), 'Destination'].fillna('TRAPPIST-1e')

    # ---------------

    # Creating a Column based on people's Journey Route: HomePlanet + Destination
    test['JourneyRoute'] = test.HomePlanet + ' to ' + test.Destination

    # ------------------------------------

    # Let's check if there is a relation between Journey Route and VIP status with the Prices paid
    # Average Room Service Price for each JourneyRouse and VIP status
    AvgRoom = np.ceil(test.groupby(['JourneyRoute', 'VIP'])['RoomService'].mean().unstack().replace(0, np.nan))

    # Filling missing RoomService values by their Journey Route and VIP's status mean (LONG VERSION)
    for i in range(AvgRoom.index.shape[0]):
        for j in range(2):
            test.loc[(test.RoomService.isna() & (test.JourneyRoute == AvgRoom.index[i]) & (test.VIP == j)),
                     'RoomService'] = test.loc[(test.RoomService.isna() & (test.JourneyRoute == AvgRoom.index[i])
                                                & (test.VIP == j)), 'RoomService'].fillna(AvgRoom[j][AvgRoom.index[i]])

    # ------

    # Average FoodCourt Price for each JourneyRouse and VIP status
    AvgFood = np.ceil(test.groupby(['JourneyRoute', 'VIP'])['FoodCourt'].mean().unstack().replace(0, np.nan))

    # Filling missing FoodCourt values by their Journey Route and VIP's status mean (LONG VERSION)
    for i in range(AvgFood.index.shape[0]):
        for j in range(2):
            test.loc[(test.FoodCourt.isna() & (test.JourneyRoute == AvgFood.index[i]) & (test.VIP == j)),
                     'FoodCourt'] = test.loc[(test.FoodCourt.isna() & (test.JourneyRoute == AvgFood.index[i])
                                              & (test.VIP == j)), 'FoodCourt'].fillna(AvgFood[j][AvgFood.index[i]])

    # ------

    # Average ShoppingMall Price for each JourneyRouse and VIP status
    AvgShopping = np.ceil(test.groupby(['JourneyRoute', 'VIP'])['ShoppingMall'].mean().unstack().replace(0, np.nan))

    # Filling missing ShoppingMall values by their Journey Route and VIP's status mean (LONG VERSION)
    for i in range(AvgShopping.index.shape[0]):
        for j in range(2):
            test.loc[(test.ShoppingMall.isna() & (test.JourneyRoute == AvgShopping.index[i]) & (test.VIP == j)),
                     'ShoppingMall'] = test.loc[(test.ShoppingMall.isna() & (test.JourneyRoute == AvgShopping.index[i])
                                                 & (test.VIP == j)), 'ShoppingMall'] \
                .fillna(AvgShopping[j][AvgShopping.index[i]])

    # ------

    # Average Spa Price for each JourneyRouse and VIP status
    AvgSpa = np.ceil(test.groupby(['JourneyRoute', 'VIP'])['Spa'].mean().unstack().replace(0, np.nan))

    # Filling missing Spa values by their Journey Route and VIP's status mean (LONG VERSION)
    for i in range(AvgSpa.index.shape[0]):
        for j in range(2):
            test.loc[(test.Spa.isna() & (test.JourneyRoute == AvgSpa.index[i]) & (test.VIP == j)), 'Spa'] = \
                test.loc[(test.Spa.isna() & (test.JourneyRoute == AvgSpa.index[i]) & (test.VIP == j)), 'Spa'] \
                    .fillna(AvgSpa[j][AvgSpa.index[i]])

    # ------

    # Average VRDeck Price for each JourneyRouse and VIP status
    AvgVRDeck = np.ceil(test.groupby(['JourneyRoute', 'VIP'])['VRDeck'].mean().unstack().replace(0, np.nan))

    # Filling missing VRDeck values by their Journey Route and VIP's status mean (LONG VERSION)
    for i in range(AvgVRDeck.index.shape[0]):
        for j in range(2):
            test.loc[(test.VRDeck.isna() & (test.JourneyRoute == AvgVRDeck.index[i]) & (test.VIP == j)), 'VRDeck'] = \
                test.loc[(test.VRDeck.isna() & (test.JourneyRoute == AvgVRDeck.index[i]) & (test.VIP == j)), 'VRDeck'] \
                    .fillna(AvgVRDeck[j][AvgVRDeck.index[i]])

    # ------------------------------------

    # Creating a Total Expenses column for summation of all Expenses
    test['TotalExpenses'] = test.RoomService + test.FoodCourt + test.ShoppingMall + test.Spa + test.VRDeck

    # ------------------------------------

    test.loc[(test.TotalExpenses.isna() & (test.CryoSleep == 1)), ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa',
                                                                   'VRDeck']] = \
        test.loc[
            (test.TotalExpenses.isna() & (test.CryoSleep == 1)), ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa',
                                                                  'VRDeck']].fillna(0)

    test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = \
        test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] \
            .fillna(test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].apply(lambda x: x.mean()))

    # Update
    test['TotalExpenses'] = test.RoomService + test.FoodCourt + test.ShoppingMall + test.Spa + test.VRDeck

    # ------------------------------------

    test.CryoSleep.fillna(0, inplace=True)
    test.VIP.fillna(0, inplace=True)

    test.Age = pd.qcut(test.Age, 5, labels=['A', 'B', 'C', 'D', 'E'])

    # test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalExpenses']] = scaler.fit_transform(
    #     test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalExpenses']])

    test['InGroupNumber'] = test.PassengerId.str.split('_').str[1]
    x = test.groupby('GroupNumber').InGroupNumber.apply(lambda Z: 'Single' if Z.max() == '01' else 'Not-Single') \
        .reset_index()
    x.columns = ['GroupNumber', 'Is_Single']
    test = test.merge(x, on='GroupNumber')
    test.drop('InGroupNumber', axis=1, inplace=True)

    test['InGroupNumber'] = test.PassengerId.str.split('_').str[1]
    x = test.groupby('GroupNumber').InGroupNumber.apply(lambda Z: Z.max()).reset_index()
    x.columns = ['GroupNumber', 'GroupSize']
    test = test.merge(x, on='GroupNumber')
    test.drop('InGroupNumber', axis=1, inplace=True)
    test.GroupSize = test.GroupSize.astype(int)

    test.GroupSize = pd.cut(test.GroupSize, 3, labels=['Small', 'Medium', 'Large'])

    test['Is_Spend'] = np.where(test.TotalExpenses > 0, 'Spent', 'Not-Spent')

    test['RoomService'] = StandardScaler().fit_transform(np.array(test['RoomService']).reshape(-1, 1))
    test['FoodCourt'] = StandardScaler().fit_transform(np.array(test['FoodCourt']).reshape(-1, 1))
    test['ShoppingMall'] = StandardScaler().fit_transform(np.array(test['ShoppingMall']).reshape(-1, 1))
    test['Spa'] = StandardScaler().fit_transform(np.array(test['Spa']).reshape(-1, 1))
    test['VRDeck'] = StandardScaler().fit_transform(np.array(test['VRDeck']).reshape(-1, 1))
    test['TotalExpenses'] = StandardScaler().fit_transform(np.array(test['TotalExpenses']).reshape(-1, 1))

    test['RatioRoomService'] = (test['RoomService'] / test['TotalExpenses']).fillna(0)
    test['RatioFoodCourt'] = (test['FoodCourt'] / test['TotalExpenses']).fillna(0)
    test['RatioShoppingMall'] = (test['ShoppingMall'] / test['TotalExpenses']).fillna(0)
    test['RatioSpa'] = (test['Spa'] / test['TotalExpenses']).fillna(0)
    test['RatioVRDeck'] = (test['VRDeck'] / test['TotalExpenses']).fillna(0)

    return test


test = cleaning_test(test)

# ------------------------------------

test_encoded = test[['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService',
                     'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck', 'Side', 'TotalExpenses',
                     'Is_Spend', 'GroupSize',
                    'RatioRoomService', 'RatioFoodCourt', 'RatioShoppingMall', 'RatioSpa', 'RatioVRDeck'
                     ]]

test_encoded = pd.get_dummies(test_encoded)  # One Hot Encoder by Pandas

prediction_test = pd.DataFrame(model.predict(test_encoded))
for i in range(prediction_test.shape[0]):
    if prediction_test[0][i] >= 0.5:
        prediction_test[0][i] = True
    else:
        prediction_test[0][i] = False
submission = pd.concat([test.PassengerId, prediction_test], axis=1)
submission.columns = ['PassengerId', 'Transported']
# submission.to_csv('submission.csv', index=False)

# ----------------------------------------------------------------------------------------------------------------------


