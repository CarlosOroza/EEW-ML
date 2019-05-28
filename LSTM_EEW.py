#Package imports
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split

#Fixed random seed for reproducibility
np.random.seed(42)

#A function to generate events, train LSTM algorithm
def run_model(mag,n_steps):
  """
  Generate earthquake with a given magnitude and random location, train LSTM model
  on each station's initial acceleration waveform 

  Keyword arguments:
  mag     -- event magnitude (float)
  n_steps -- the number of timesteps to include in training (int)

  Returns:
  Pandas Dataframe of true and predicted PGA for each station
  """
  
  #Generate random epicenter 
  epicenter_lon=uniform(low=-119,high=-116)
  epicenter_lat=uniform(low=33,high=35)

  #Generate a single event
  N,E,Z=make_seismograms(t,station_lon,station_lat,station_vs30,mag,
                         epicenter_lon,epicenter_lat)

  #Convert N,E,Z components to total acceleration
  station_accel = np.sqrt(N**2 + E**2 + Z**2)

  #Find the index where the event starts in the accel data for all stations 
  station_accel_start = np.argmax(station_accel > 0, axis = 0)

  #Find the time between max PGA and event start
  t_start_accel = np.argmax(station_accel, axis=0) - station_accel_start

  #Filter any stations that see max PGA within the training window, reshape for LSTM
  station_accel = station_accel[:,t_start_accel > n_steps][np.newaxis,:,:].T

  #Create empty X feature matrix
  X = np.zeros((station_accel.shape[0],n_steps,station_accel.shape[2]))

  #Add station subsampled timeseries to X feature matrix
  for i in range(station_accel.shape[0]):
    X[i,:,:] = station_accel[i,station_accel_start[i]:
                               station_accel_start[i] + n_steps]

  #Create y target vector, the max acceleration for each station  
  y = station_accel.max(axis = 1)

  #Generate randomly subsampled training and testing data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

  #Create LSTM model (needs to be expanded to include hyperparameter tuning)
  model = Sequential()
  model.add(LSTM(100, input_shape=(X_train.shape[1],X_train.shape[2])))
  model.add(Dense(y_train.shape[1]))
  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
  model.fit(X_train, y_train, epochs=10, verbose=0)

  #Output a Pandas dataframe with the true and predicted data
  out_df = pd.DataFrame()
  out_df['True PGA'] = y_test.flatten()
  out_df['Predicted PGA'] = model.predict(X_test).flatten()
  out_df['Magnitude'] = np.full(np.shape(y_test)[0],mag)
  
  return out_df

#Initialize a Pandas dataframe
all_runs = pd.DataFrame(columns = ['True PGA','Predicted PGA','Magnitude'])

#Evaluate the LSTM algorithm with 100 events of magnitudes 3 - 5.9
for magnitude in np.linspace(3,5.9,100):
  all_runs = all_runs.append(run_model(magnitude,500))

