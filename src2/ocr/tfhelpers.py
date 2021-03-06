import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMCell, ResidualWrapper, DropoutWrapper, MultiRNNCell

class Model():
    def __init__(self, loc, operation='activation', input_name='x'):
        self.input = input_name + ":0"
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        with self.graph.as_default():
            saver = tf.compat.v1.train.import_meta_graph(loc + '.meta', clear_devices=True)
            saver.restore(self.sess, loc)
            self.op = self.graph.get_operation_by_name(operation).outputs[0]
    def run(self, data):
        return self.sess.run(self.op, feed_dict={self.input: data})
    def eval_feed(self, feed):
        return self.sess.run(self.op, feed_dict=feed)
    def run_op(self, op, feed, output=True):
        if output:
            return self.sess.run(self.graph.get_operation_by_name(op).outputs[0], feed_dict=feed)
        else:
            self.sess.run(self.graph.get_operation_by_name(op), feed_dict=feed)
        
    
    
def _create_single_cell(cell_fn, num_units, is_residual=False, is_dropout=False, keep_prob=None):
    cell = cell_fn(num_units)
    if is_dropout:
        cell = DropoutWrapper(cell, input_keep_prob=keep_prob)
    if is_residual:
        cell = ResidualWrapper(cell)
    return cell


def create_cell(num_units, num_layers, num_residual_layers, is_dropout=False, keep_prob=None, cell_fn=LSTMCell):
    cell_list = []
    for i in range(num_layers):
        cell_list.append(_create_single_cell(cell_fn=cell_fn,
            num_units=num_units, is_residual=(i >= num_layers - num_residual_layers),
            is_dropout=is_dropout, keep_prob=keep_prob
        ))
    if num_layers == 1:
        return cell_list[0]
    return MultiRNNCell(cell_list)