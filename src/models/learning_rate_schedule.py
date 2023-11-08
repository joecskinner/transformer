import tensorflow as tf


class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps):
        super(CustomLearningRateSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step_num):
        step_num = tf.cast(step_num, tf.float32)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(tf.math.rsqrt(step_num), step_num * (self.warmup_steps ** -1.5))
