#encoding=utf-8

import sys
import tensorflow as tf

def load_model_meta_ckpt(meta_path, ckpt_path):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(sess, ckpt_path)
    return sess

def save_model(session, export_dir):
    inputs = {
            "input_ids":
            session.graph.get_tensor_by_name("Placeholder:0"),
            "input_mask":
            session.graph.get_tensor_by_name("Placeholder_1:0"),
            "input_type_ids":
            session.graph.get_tensor_by_name("Placeholder_2:0"),
            }
    outputs = {
            "layer_"+str(i):
            session.graph.get_tensor_by_name("bert/encoder/Reshape_{}:0".format(i+1)) for i in range(12)
            }
    tf.saved_model.simple_save(
            session,
            export_dir,
            inputs,
            outputs,
            )


def list_tensor(graph):
    for n in graph.as_graph_def().node:
        print(n.name)

def main(pretrain_path, save_path):
    meta_path = pretrain_path+'/bert_model.ckpt.meta'
    ckpt_path = pretrain_path+'/bert_model.ckpt'
    sess = load_model_meta_ckpt(meta_path, ckpt_path)
    # list_tensor(sess.graph)
    save_model(sess, save_path)

main(sys.argv[1], sys.argv[2])
