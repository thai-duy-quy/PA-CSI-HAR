from tensorflow.keras import layers
from tensorflow.keras.initializers import GlorotUniform
import numpy as np

class LayerNorm(layers.Layer):
    def __init__(self,features,eps=1e-6):
        super().__init__()
        self.a_2 = np.ones(features)
        self.b_2 = np.ones(features)
        self.eps = eps
    
    def call(self,x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        out = self.a_2 * (x-mean)/(std + self.eps) + self.b_2
        return out

class Encoder(layers.Layer):
    def __init__(self,layer,N):
        super().__init__()
        self.layers = []
        for i in range(N):
            self.layers.append(layer)
        self.norm = LayerNorm(500) #layer.output_shape) #need check 
    def call(self,x, mask = None):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class SublayerConnection(layers.Layer):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = layers.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(layers.Layer):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer1 = SublayerConnection(size, dropout)
        self.sublayer2 = SublayerConnection(size, dropout)
        self.size = size

    def forward(self, x, mask=None):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    d_k = tf.shape(query)[-1]
    scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.cast(d_k, tf.float32))
    if mask is not None:
        scores = tf.where(tf.equal(mask, 0), tf.fill(tf.shape(scores), -1e9), scores)
    p_attn = tf.nn.softmax(scores, axis=-1)
    if dropout is not None:
        p_attn = Dropout(dropout)(p_attn, training=True)
    return tf.matmul(p_attn, value), p_attn

class PositionwiseFeedForward(layers.Layer):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = layers.Dense(d_ff, activation='relu', kernel_initializer=GlorotUniform())
        self.w_2 = layers.Dense(d_model, kernel_initializer=GlorotUniform())
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        x = self.w_1(x)
        x = self.dropout(x, training=training)
        return self.w_2(x)

class MultiHeadedAttention(layers.Layer):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = []
        for i in range(4):
            self.linears.append(layers.Dense(d_model))
        self.attn = None
        self.dropout = layers.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = tf.expand_dims(mask, axis=1)
        nbatches = tf.shape(query)[0]

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            tf.transpose(tf.reshape(l(x), (nbatches, -1, self.h, self.d_k)), perm=[0, 2, 1, 3])
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask,
                                      dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]), (nbatches, -1, self.h * self.d_k))

        return self.linears[-1](x)    

class Transfomer(layers.Layer):
    def __init__(self,hidden_dim,N,H):
        super().__init__()
        self.model = Encoder(
            EncoderLayer(hidden_dim,MultiHeadedAttention(H,hidden_dim),
            PositionwiseFeedForward(hidden_dim, hidden_dim*4),
            0.1
            ),
            N
        )
    def forward(self,x,mask=None):
        self.model(x,mask)