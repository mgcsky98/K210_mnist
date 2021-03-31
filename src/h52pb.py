#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-02-22 10:59
# @Author  : MGC
# @Site    : 
# @File    : h52pb.py
# @Software: PyCharm
import tensorflow as tf
import tensorflow.compat.v1 as tf1

tf1.reset_default_graph()
tf1.keras.backend.set_learning_phase(0)  # 调用模型前一定要执行该命令
tf1.disable_v2_behavior()  # 禁止tensorflow2.0的行为
# 加载hdf5模型
hdf5_pb_model = tf.keras.models.load_model("mnist_lenet_K210_model.h5")


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        #         freeze_var_names = list(set(v.op.name for v in tf1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        #         output_names += [v.op.name for v in tf1.global_variables()]
        print("output_names", output_names)
        input_graph_def = graph.as_graph_def()
        #         for node in input_graph_def.node:
        #             print('node:', node.name)
        print("len node1", len(input_graph_def.node))
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                                     output_names)

        outgraph = tf1.graph_util.remove_training_nodes(frozen_graph)  # 云掉与推理无关的内容
        print("##################################################################")
        for node in outgraph.node:
            print('node:', node.name)
        print("len node1", len(outgraph.node))
        return outgraph


frozen_graph = freeze_session(tf1.keras.backend.get_session(),
                              output_names=[out.op.name for out in hdf5_pb_model.outputs])
tf1.train.write_graph(frozen_graph, '', "mnist_lenet_K210_model.pb", as_text=False)

