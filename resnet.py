import tensorflow as tf

class SimpleResNetModule(tf.keras.layers.Layer):
    """
    Initializes a single resnet block:
    
            ---------------------
    X _____|                     |_____ Conv2D --> Y
           |___ Conv2D (same) ___|

    """
    
    def __init__(
        self,
        name = None,
        hparams = {"filters": 16, "kernel_size": 3, "activation": "relu"}
    ):
        super().__init__(name = name)
        self.hparams = hparams

    def build(self, input_shape):
        self.inner_conv2d = tf.keras.layers.Conv2D(
            **self.hparams,
            padding = "same",
            strides = 1,
            input_shape = input_shape[1:]
        )
        self.outer_conv2d = tf.keras.layers.Conv2D(
            **self.hparams,
            padding = "same",
            strides = 2,
            input_shape = input_shape[1:]
        )

    @tf.function
    def call(self, x):
        x_inner = self.inner_conv2d(x)
        x_combined = tf.concat((x, x_inner), axis = -1)
        
        y = self.outer_conv2d(x_combined)
        return y
   
