import tensorflow as tf
import numpy as np

class Converter(object):

    def __init__(self, tf_nodes, mx_nodes, mx_params):
        self.tf_nodes = tf_nodes
        self.mx_nodes = mx_nodes
        self.mx_params = mx_params

    def to_tuple(self, string, conv_type=str):
        return tuple(map(conv_type, map(str.strip, string[1:-1].split(','))))

    def create_var(self, node, shape=None):
        node_name = node['name']
        if shape is None:
            if node_name in self.mx_params:
                shape = self.mx_params[node_name].shape
            else:
                shape = ()
        # print('Creating var with shape:', shape)
        created_node = tf.get_variable(node_name, shape=shape, initializer=tf.zeros_initializer, trainable = False)
        self.tf_nodes[node_name] = created_node
        return created_node

    def create_bn(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]

        epsilon = float(node['attrs']['eps'])
        input_shape = input_sym.get_shape()
        axis = list(range(len(input_shape) - 1))

        def create_bn_params(i):
            cur_node = self.mx_nodes[node['inputs'][i][0]]
            cur_name = cur_node['name']
            self.create_var(cur_node)
            self.tf_nodes[cur_name].load(self.mx_params[cur_name].asnumpy())
            return self.tf_nodes[cur_name]
        if len(node['inputs']) > 3:
            gamma, beta, mean, var = (create_bn_params(i) for i in range(1, 5))
        else:
            gamma, beta = (create_bn_params(i) for i in range(1, 3))
            mean = tf.get_variable(node_name + '_mean', shape=input_shape[-1], initializer=tf.zeros_initializer, trainable = False)
            mean.load(np.zeros((input_shape[-1],), dtype='float32'))
            var = tf.get_variable(node_name + '_var', shape=input_shape[-1], initializer=tf.ones_initializer, trainable = False)
            var.load(np.ones((input_shape[-1],), dtype='float32'))
        if 'fix_gamma' in node['attrs']:
            if node['attrs']['fix_gamma'] == 'True':
                # print('Fix')
                gamma = tf.get_variable(node_name + '_gamma_fixed', shape=input_shape[-1], initializer=tf.ones_initializer, trainable = False)
                gamma.load(np.ones((input_shape[-1],), dtype='float32'))
        else:
            gamma = tf.get_variable(node_name + '_gamma_fixed', shape=input_shape[-1], initializer=tf.ones_initializer, trainable = False)
            gamma.load(np.ones((input_shape[-1],), dtype='float32'))
        self.tf_nodes[node_name] = tf.nn.batch_normalization(input_sym, mean, var, beta, gamma, epsilon, name=node_name)
        return self.tf_nodes[node_name]

    def create_conv(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        num_filters_in = input_sym.get_shape()[-1]
        num_filters_out = int(node['attrs']['num_filter'])
        kernel_size = self.to_tuple(node['attrs']['kernel'], int)

        if 'no_bias' in node['attrs']:
            if node['attrs']['no_bias']:
                add_bias = False
            else:
                add_bias = True
        else:
            # by default, bias exists
            add_bias = True
                   
        # add bias
        if add_bias:
            bias_node = self.mx_nodes[node['inputs'][2][0]]
            #print("-----------------------conv name:", node_name, ", bias:", add_bias, "name:", bias_node['name'])
            bias = self.create_var(bias_node, shape=(num_filters_out))
            #print("bias shape:", bias.shape)
            bias_numpy = self.mx_params[bias_node['name']].asnumpy()
            bias.load(bias_numpy)
        
        if 'num_group' in node['attrs']:
            num_group = int(node['attrs']['num_group'])
        else:
            num_group = 1
        if 'pad' in node['attrs']:
            padding = self.to_tuple(node['attrs']['pad'], int)
        else:
            padding = (0, 0)
        stride = self.to_tuple(node['attrs']['stride'], int)
        
        weights_node = self.mx_nodes[node['inputs'][1][0]]
        weights_numpy = self.mx_params[weights_node['name']].asnumpy().transpose((2, 3, 1, 0))
        
        if padding[0] > 0 or padding[1] > 0:
            padded_input = tf.pad(input_sym, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]], 'CONSTANT')
        else:
            padded_input = input_sym
        convolve = lambda input_sym, kernel, name=None: tf.nn.conv2d(input_sym, kernel, [1, stride[0], stride[1], 1], padding='VALID', name=name)
        
        if num_group > 1:
            #redefine with group conv.
            weights = self.create_var(weights_node,
                                 shape=(kernel_size[0], kernel_size[1], num_filters_in, 1))
            weights.load(weights_numpy.transpose((0,1,3,2)))
            
            self.tf_nodes[node_name] = tf.nn.depthwise_conv2d(padded_input, weights, strides = [1, stride[0], stride[1], 1], padding='VALID', name = node_name)
        else:
            weights = self.create_var(weights_node,
                                 shape=(kernel_size[0], kernel_size[1], num_filters_in // num_group, num_filters_out))
            weights.load(weights_numpy)
            if add_bias:
                _tmp_node = convolve(padded_input, weights, name=node_name+"_before_bias")
                self.tf_nodes[node_name] = tf.add(_tmp_node, bias, name = node_name)
            else:
                self.tf_nodes[node_name] = convolve(padded_input, weights, name=node_name)

        return self.tf_nodes[node_name]
    
    def create_fc(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        #if node is not 1-d vector, we flatten it.
        if len(input_sym.get_shape()) > 2:
            input_sym = tf.layers.flatten(input_sym)
        
        num_units_in = input_sym.get_shape()[1]
        num_units_out = int(node['attrs']['num_hidden'])
        weights_node = self.mx_nodes[node['inputs'][1][0]]
        weights = self.create_var(weights_node, shape=(num_units_in, num_units_out))
        bias_node = self.mx_nodes[node['inputs'][2][0]]
        bias = self.create_var(bias_node, shape=(num_units_out,))
        weights_numpy = self.mx_params[weights_node['name']].asnumpy()
        weights.load(weights_numpy.T)
        bias.load(self.mx_params[bias_node['name']].asnumpy())
        self.tf_nodes[node_name] = tf.nn.xw_plus_b(input_sym, weights, bias, name=node_name)
        return self.tf_nodes[node_name]
    
    def create_pooling(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        pooling_type = node['attrs']['pool_type']
        kernel_size = self.to_tuple(node['attrs']['kernel'], int)
        if 'stride' in node['attrs']:
            stride = self.to_tuple(node['attrs']['stride'], int)
        else:
            stride = (1, 1)
        if 'global_pool' in node['attrs']:
            global_pool = node['attrs']['global_pool'] == 'True'
        else:
            global_pool = False
        if 'pad' in node['attrs']:
            padding = self.to_tuple(node['attrs']['pad'], int)
        else:
            padding = (0, 0)
        if global_pool:
            self.tf_nodes[node_name] = tf.reduce_mean(input_sym, reduction_indices=[1, 2], name=node_name)
        else:
            if padding[0] > 0 or padding[1] > 0:
                padded_input = tf.pad(input_sym,
                                      [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]],
                                      'CONSTANT')
            else:
                padded_input = input_sym
            if pooling_type == 'max':
                self.tf_nodes[node_name] = tf.nn.max_pool(padded_input,
                                                          ksize=[1, kernel_size[0], kernel_size[1], 1],
                                                          strides=[1, stride[0], stride[1], 1],
                                                          padding='VALID', name=node_name)
            else:
                raise NameError('Unknown pooling type: %s' % pooling_type)
        return self.tf_nodes[node_name]

    def create_activation(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        activation_type = node['attrs']['act_type']
        if activation_type == 'relu':
            activation_fn = tf.nn.relu
        else:
            raise NameError('Unknown activation type: %s' % activation_type)
        self.tf_nodes[node_name] = activation_fn(input_sym, name=node_name)
        return self.tf_nodes[node_name]
    
    def create_lrelu(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        
        act_type = node['attrs']['act_type']
        #print("act type:", act_type)
        
        alpha_sym = self.mx_nodes[node['inputs'][1][0]]
        alpha_name = alpha_sym['name']
        alpha = self.mx_params[alpha_name].asnumpy()
        
        pos = tf.nn.relu(input_sym)
        neg = tf.multiply( tf.multiply( tf.subtract(input_sym, abs(input_sym)), 0.5 ), alpha )
        self.tf_nodes[node_name] = tf.add(pos, neg, name=node_name)
        
        return self.tf_nodes[node_name]
    
    def create_concat(self, node):
        node_name = node['name']
        input_nodes = node['inputs']
        input_syms = []
        for i in range(len(node['inputs'])):
            _node = node['inputs'][i][0]
            input_syms.append(self.tf_nodes[self.mx_nodes[node['inputs'][i][0]]['name']])
        self.tf_nodes[node_name] = tf.concat(input_syms, axis = 3, name=node_name)
        return self.tf_nodes[node_name]
    
    def create_reshape(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        #print("reshape input:", input_sym)
        resize_shape = node['attrs']['shape'] #NCHW
        resize_shape = resize_shape.strip(')')
        resize_shape = resize_shape.strip('(')
        resize_shape = resize_shape.split(',')
        for i in range(len(resize_shape)):
            resize_shape[i] = int(resize_shape[i])
        #print("arg:", resize_shape)
        resize_shape_tf = [resize_shape[0], resize_shape[2], resize_shape[3], resize_shape[1]] #NHWC

        resize_shape_tf[0] = 1 # batch dimension
        for i in range(1, len(resize_shape_tf)):
            if resize_shape_tf[i] == 0:
                resize_shape_tf[i] = input_sym.shape[i].value #can also work with input_sym.get_shape().as_list()
        #print("reshape:", resize_shape_tf)

        self.tf_nodes[node_name] = tf.reshape(input_sym, shape = resize_shape_tf, name=node_name)
        #print("resized output:", self.tf_nodes[node_name])
        return self.tf_nodes[node_name]
        
    def create_softmaxactivation(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        print("softmax input:", input_sym)

        self.tf_nodes[node_name] = tf.nn.softmax(input_sym, axis = 3, name=node_name)
        return self.tf_nodes[node_name]
    
    def create_dropout(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        # at test time, we ignore dropout.
        self.tf_nodes[node_name] = tf.identity(input_sym, name=node_name)
        #self.tf_nodes[node_name] = tf.nn.dropout(input_sym, keep_prob = 0.6, name=node_name)
        return self.tf_nodes[node_name]

    def create_copy(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        self.tf_nodes[node_name] = tf.identity(input_sym, name=node_name)

        return self.tf_nodes[node_name]
    
    def create_minus(self, node):
        node_name = node['name']
        val = float(node['attrs']["scalar"])
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        self.tf_nodes[node_name] = tf.add(input_sym, -1*val, name=node_name.strip("_"))

        return self.tf_nodes[node_name]
    
    def create_multiply(self, node):
        node_name = node['name']
        val = float(node['attrs']["scalar"])
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        self.tf_nodes[node_name] = tf.multiply(input_sym, val, name=node_name.strip("_"))

        return self.tf_nodes[node_name]
    
    def create_crop(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        target_sym = self.tf_nodes[self.mx_nodes[node['inputs'][1][0]]['name']]
        self.tf_nodes[node_name] = tf.slice(input_sym, [0,0,0,0], [-1,target_sym.shape[1], target_sym.shape[2], -1], name=node_name)
        return self.tf_nodes[node_name]
            
    def create_upsampling(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        scale = node['attrs']['scale']
        new_size = [input_sym.shape[1]*scale, input_sym.shape[2]*scale]

        # defined in https://www.tensorflow.org/api_docs/python/tf/image/resize_nearest_neighbor
        self.tf_nodes[node_name] = tf.image.resize_nearest_neighbor(input_sym, size = new_size, name=node_name)
        return self.tf_nodes[node_name]
    
    def create_upsampling_v2(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        #print("input to upsampling:", input_sym)
        scale = int(node['attrs']['scale'])
        new_size = [1, input_sym.shape[1]*scale, input_sym.shape[2]*scale, input_sym.shape[3]]

        
        def tf_repeat(tensor, repeats, name):
            '''
            expanded_tensor = tf.expand_dims(tensor, -1)
            multiples = [1] + repeats
            tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
            #tiled_tensor = tf.squeeze(tiled_tensor, axis = 4)
            tiled_tensor = tf.squeeze(tiled_tensor)
            repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats, name = name)
            '''

            #'''
            # this is the revised version that does not require expanding tensor to 5-dim
            tiled_tensor = tf.tile(tensor, multiples = [1]+repeats[0:3])
            repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats, name = name)
            #'''
            return repeated_tesnor
        
        self.tf_nodes[node_name] = tf_repeat(input_sym, [1, scale, scale, 1], name=node_name)
        print("#######################upsampling out node:", self.tf_nodes[node_name])
        return self.tf_nodes[node_name]
        
    def create_softmax(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        self.tf_nodes[node_name] = tf.nn.softmax(input_sym, name=node_name)
        return self.tf_nodes[node_name]

    def create_elementwise(self, node, op='sum'):
        node_name = node['name']
        inputs_sym = [self.tf_nodes[self.mx_nodes[n[0]]['name']] for n in node['inputs']]

        if op == 'sum':
            if len(inputs_sym) == 2:
                print("-------------adding just 2 nodes, replace with tf.add-------------------")
                self.tf_nodes[node_name] = tf.add(inputs_sym[0], inputs_sym[1], name=node_name.strip("_"))
            else:
                self.tf_nodes[node_name] = tf.add_n(inputs_sym, name=node_name.strip("_"))
            
        else:
            raise NameError('Unknown elementwise type: %s' % op)
        return self.tf_nodes[node_name]

    def create_norm(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        self.tf_nodes[node_name] = tf.nn.l2_normalize(input_sym, dim=1, name=node_name)
        return self.tf_nodes[node_name]

    def create_flatten(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        self.tf_nodes[node_name] = tf.contrib.layers.flatten(input_sym)
        return self.tf_nodes[node_name]
