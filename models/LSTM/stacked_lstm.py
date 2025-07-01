from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

class StackedLSTM:
    def __init__(self, window_size, num_features, 
                lstm_units=[50, 60, 80, 120], 
                dropout_rate=[0.3, 0.3, 0.3, 0.3], 
                loss='mse', optimizer='adam'): 
        self.window_size = window_size
        self.num_features = num_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.loss = loss
        self.optimizer = optimizer
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()

        model.add(LSTM(self.lstm_units[0], 
                    activation='relu',
                    return_sequences=True, 
                    input_shape=(self.window_size, self.num_features)))
        model.add(Dropout(self.dropout_rate[0]))

        model.add(LSTM(self.lstm_units[1], 
                    activation='relu',
                    return_sequences=True))
        model.add(Dropout(self.dropout_rate[1]))

        model.add(LSTM(self.lstm_units[2], 
                    activation='relu',
                    return_sequences=True))
        model.add(Dropout(self.dropout_rate[2]))

        model.add(LSTM(self.lstm_units[3], 
                    activation='relu'))
        model.add(Dropout(self.dropout_rate[3]))
        
        model.add(Dense(1))
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model
    
    def fit(self, X_train, y_train, epochs=100, batch_size=64, validation_data=None, verbose=1, **kwargs):
        return self.model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=validation_data,
            verbose=verbose,
            **kwargs
        )
        
    def predict(self, X):
        return self.model.predict(X)
        
    def summary(self):
        self.model.summary()

    def save(self, filepath):
        self.model.save(filepath)
        
    def load(self, filepath):
        self.model = load_model(filepath)