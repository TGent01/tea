from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense
from keras.regularizers import l2 as l2_reg_fn

SEQUENCE_LENGTH = 40
FEATURE_DIM = 64  # debe coincidir con la salida del extractor CNN (cnn_extractor.py)


def crear_lstm_clasificador(num_clases, feature_dim=FEATURE_DIM, dropout=0.3, l2_reg=0.0):
    """Sub-modelo temporal (capa LSTM), completamente independiente de la CNN.

    Recibe una secuencia de vectores de características (uno por frame, de
    forma (SEQUENCE_LENGTH, feature_dim)) y produce la clasificación final
    entre las clases de conducta.

    Parámetros (para poder comparar configuraciones de forma controlada,
    ver entrenar_multiples_semillas.py):
        dropout: probabilidad de Dropout aplicada después de la LSTM.
        l2_reg: fuerza de la regularización L2 sobre los pesos de la LSTM
            (kernel y recurrente) y de la capa de salida. 0.0 la desactiva.
    """
    entrada = Input(shape=(SEQUENCE_LENGTH, feature_dim), name="secuencia_caracteristicas")
    regularizador = l2_reg_fn(l2_reg) if l2_reg > 0 else None

    x = LSTM(
        32, name="lstm_temporal",
        kernel_regularizer=regularizador,
        recurrent_regularizer=regularizador,
    )(entrada)
    x = Dropout(dropout, name="dropout_temporal")(x)
    salida = Dense(num_clases, activation="softmax", kernel_regularizer=regularizador, name="clasificacion")(x)

    return Model(inputs=entrada, outputs=salida, name="LSTM_clasificador_temporal")


if __name__ == "__main__":
    clasificador = crear_lstm_clasificador(num_clases=3)
    clasificador.summary()
