import jax
import jax.numpy as jnp
import onnx
from onnx import helper, TensorProto
import numpy as np
from typing import Dict, List, Tuple, Any, Set, Optional

class JaxprToOnnx:
    """
    A translator that converts JAX's JAXPR representation to ONNX format.
    """
    def __init__(self, name_counter=0):
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.initializers = []
        self.value_info = []
        self._name_counter_init = name_counter
        self.name_counter = name_counter
        self.var_to_name: Dict[Any, str] = {}
        self.primitive_handlers = {
            jax.lax.neg_p: self._handle_neg,
            jax.lax.not_p: self._handle_not,
            jax.lax.add_p: self._handle_add,
            jax.lax.mul_p: self._handle_mul,
            jax.lax.sub_p: self._handle_sub,
            jax.lax.div_p: self._handle_div,
            jax.lax.and_p: self._handle_and,
            jax.lax.or_p: self._handle_or,
            jax.lax.xor_p: self._handle_xor,
            jax.lax.eq_p: self._handle_eq,
            jax.lax.ne_p: self._handle_ne,
            jax.lax.lt_p: self._handle_lt,
            jax.lax.gt_p: self._handle_gt,
            jax.lax.max_p: self._handle_max,
            jax.lax.min_p: self._handle_min,
            jax.lax.select_n_p: self._handle_select_n,
            jax.lax.dot_general_p: self._handle_dot_general,
            jax.lax.reduce_sum_p: self._handle_reduce_sum,
            jax.lax.reduce_max_p: self._handle_reduce_max,
            jax.lax.reduce_min_p: self._handle_reduce_min,
            jax.lax.gather_p: self._handle_gather,
            jax.lax.scatter_add_p: self._handle_scatter_add, # TODO: CHANGE
            jax.lax.argmax_p: self._handle_argmax,
            jax.lax.argmin_p: self._handle_argmin,
            jax.lax.square_p: self._handle_square,
            jax.lax.integer_pow_p: self._handle_integer_pow,
            jax.lax.sqrt_p: self._handle_sqrt,
            jax.lax.exp_p: self._handle_exp,
            jax.lax.log_p: self._handle_log,
            jax.lax.tanh_p: self._handle_tanh,
            jax.lax.iota_p: self._handle_iota,
            # jax.lax.sigmoid_p: self._handle_sigmoid,
            jax.lax.reshape_p: self._handle_reshape,
            jax.lax.transpose_p: self._handle_transpose,
            jax.lax.squeeze_p: self._handle_squeeze,
            jax.lax.broadcast_in_dim_p: self._handle_broadcast_in_dim,
            jax.lax.slice_p: self._handle_slice,
            jax.lax.concatenate_p: self._handle_concatenate,
            jax.lax.conv_general_dilated_p: self._handle_conv,
            jax.lax.sort_p: self._handle_sort,
            jax.lax.stop_gradient_p: self._handle_stop_gradient, 
            jax._src.prng.random_seed_p: self._handle_random_seed, 
            jax._src.prng.random_wrap_p: self._handle_random_wrap, 
            jax._src.prng.random_split_p: self._handle_random_split,
            jax._src.prng.random_unwrap_p: self._handle_random_unwrap,
            jax.lax.convert_element_type_p: self._handle_convert_element_type, # TODO: CHANGE
            jax.lax.device_put_p: self._handle_device_put,
            jax.random.random_gamma_p: self._handle_random_gamma,
        }
    
    def _get_unique_name(self, prefix="node"):
        """Generate a unique name for ONNX nodes."""
        name = f"{prefix}_{self.name_counter}"
        self.name_counter += 1
        return name
    
    def _get_var_name(self, var):
        """Get or create a name for a JAX variable."""
        if var not in self.var_to_name:
            self.var_to_name[var] = self._get_unique_name(f"var")
        return self.var_to_name[var]
    
    def _get_constant_name(self, val):
        """Add a constant to the model and return its name."""
        name = self._get_unique_name("const")
        # Convert to numpy and create tensor
        if isinstance(val, jax._src.core.Literal):
            actual_val = val.val
        else:
            actual_val = val
        
        np_val = np.array(actual_val)
        if np_val.dtype == np.float64:
            np_val = np_val.astype(np.float32)

        tensor = helper.make_tensor(
            name=name,
            data_type=self._numpy_dtype_to_onnx(np_val.dtype),
            dims=np_val.shape,
            vals=np_val.flatten().tolist(),
        )
        self.initializers.append(tensor)
        return name
    
    def _numpy_dtype_to_onnx(self, dtype):
        """Convert numpy dtype to ONNX data type."""
        if dtype == np.float32:
            return TensorProto.FLOAT
        elif dtype == np.float64:
            return TensorProto.DOUBLE
        elif dtype == np.int32:
            return TensorProto.INT32
        elif dtype == np.int64:
            return TensorProto.INT64
        elif dtype == np.bool_:
            return TensorProto.BOOL
        else:
            return TensorProto.FLOAT
    
    def _add_input(self, var, shape, dtype=np.float32):
        """Add an input to the ONNX model."""
        name = self._get_var_name(var)
        input_def = helper.make_tensor_value_info(
            name, self._numpy_dtype_to_onnx(dtype), shape
        )
        self.inputs.append(input_def)
        return name
    
    def _add_output(self, var, shape, dtype=np.float32):
        """Add an output to the ONNX model."""
        name = self._get_var_name(var)
        output_def = helper.make_tensor_value_info(
            name, self._numpy_dtype_to_onnx(dtype), shape
        )
        self.outputs.append(output_def)
        return name
    
    def _add_intermediate(self, var, shape, dtype=np.float32):
        """Add an intermediate value to the ONNX model."""
        name = self._get_var_name(var)
        value_info = helper.make_tensor_value_info(
            name, self._numpy_dtype_to_onnx(dtype), shape
        )
        self.value_info.append(value_info)
        return name
    
    def _get_name(self, var):
        if isinstance(var, jax._src.core.Var):
            return self._get_var_name(var)
        elif isinstance(var, jax._src.core.Literal):
            return self._get_constant_name(var)
        else:
            raise NotImplementedError("not yet implemented")

    def _handle_identity(self, node_inputs, node_outputs, params):
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])   

        node = helper.make_node(
            "Identity",
            inputs = input_names,
            outputs = [output_name],
            name = self._get_unique_name("identity")
        )
        self.nodes.append(node)

    def _handle_stop_gradient(self, node_inputs, node_outputs, params):
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])   

        node = helper.make_node(
            "Identity",
            inputs = input_names,
            outputs = [output_name],
            name = self._get_unique_name("stop_gradient")
        )
        self.nodes.append(node)

    def _handle_random_seed(self, node_inputs, node_outputs, params):
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])   

        node = helper.make_node(
            "Identity",
            inputs = input_names,
            outputs = [output_name],
            name = self._get_unique_name("random_seed")
        )
        self.nodes.append(node)

    def _handle_random_wrap(self, node_inputs, node_outputs, params):
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])   

        node = helper.make_node(
            "Identity",
            inputs = input_names,
            outputs = [output_name],
            name = self._get_unique_name("random_wrap")
        )
        self.nodes.append(node)

    def _handle_random_split(self, node_inputs, node_outputs, params):
        input_name = self._get_name(node_inputs[0])
        intermediate = self._get_unique_name("random_split:x")
        output_name = self._get_var_name(node_outputs[0])   

        reshape = self._get_constant_name(np.array([1, 2], dtype=np.int64))

        num = params['shape'][0]
        repeat = self._get_constant_name(np.array([num,1], dtype=np.int64))


        node_1 = helper.make_node(
            "Reshape", 
            inputs = [input_name, reshape],
            outputs = [intermediate],
            name = self._get_unique_name("random_split:reshape")
        )
        self.nodes.append(node_1)

        node_2 = helper.make_node(
            "Tile",
            inputs = [intermediate, repeat],
            outputs = [output_name],
            name = self._get_unique_name("random_split:tile")
        )
        self.nodes.append(node_2)

    def _handle_random_unwrap(self, node_inputs, node_outputs, params):
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])   

        node = helper.make_node(
            "Identity",
            inputs = input_names,
            outputs = [output_name],
            name = self._get_unique_name("random_wrap")
        )
        self.nodes.append(node)

    def _handle_neg(self, node_inputs, node_outputs, params):
        """Handle JAX neg primitive."""
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])
        node = helper.make_node(
            "Neg",
            inputs=input_names,
            outputs=[output_name],
            name=self._get_unique_name("neg")
        )
        self.nodes.append(node)

    def _handle_not(self, node_inputs, node_outputs, params):
        """Handle JAX not primitive."""
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])
        node = helper.make_node(
            "Not",
            inputs=input_names,
            outputs=[output_name],
            name=self._get_unique_name("not")
        )
        self.nodes.append(node)
        
    
    def _handle_add(self, node_inputs, node_outputs, params):
        """Handle JAX add primitive."""
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])
        node = helper.make_node(
            "Add",
            inputs=input_names,
            outputs=[output_name],
            name=self._get_unique_name("add")
        )
        self.nodes.append(node)
    
    def _handle_mul(self, node_inputs, node_outputs, params):
        """Handle JAX mul primitive."""
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])
        node = helper.make_node(
            "Mul",
            inputs=input_names,
            outputs=[output_name],
            name=self._get_unique_name("mul")
        )
        self.nodes.append(node)
    
    def _handle_sub(self, node_inputs, node_outputs, params):
        """Handle JAX sub primitive."""
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])
        node = helper.make_node(
            "Sub",
            inputs=input_names,
            outputs=[output_name],
            name=self._get_unique_name("sub")
        )
        self.nodes.append(node)
    
    def _handle_div(self, node_inputs, node_outputs, params):
        """Handle JAX div primitive."""
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])
        node = helper.make_node(
            "Div",
            inputs=input_names,
            outputs=[output_name],
            name=self._get_unique_name("div")
        )
        self.nodes.append(node)

    def _handle_eq(self, node_inputs, node_outputs, params):
        """Handle JAX eq primitive"""
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])
        node = helper.make_node(
            "Equal",
            inputs = input_names,
            outputs=[output_name],
            name=self._get_unique_name("eq")
        )
        self.nodes.append(node)

    def _handle_ne(self, node_inputs, node_outputs, params):
        """Handle JAX ne primitive"""
        input_names = [self._get_name(inp) for inp in node_inputs]
        eq_output = self._get_unique_name("equal_output")
        output_name = self._get_var_name(node_outputs[0])
        node_1 = helper.make_node(
            "Equal",
            inputs = input_names,
            outputs=[eq_output],
            name=self._get_unique_name("ne_eq")
        )
        self.nodes.append(node_1)

        node_2 = helper.make_node(
            "Not",
            inputs = [eq_output],
            outputs=[output_name],
            name=self._get_unique_name("ne_not")
        )
        self.nodes.append(node_2)

    def _handle_and(self, node_inputs, node_outputs, params):
        """Handle JAX and primitive."""
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])
        node = helper.make_node(
            "And",
            inputs=input_names,
            outputs=[output_name],
            name=self._get_unique_name("and")
        )
        self.nodes.append(node)

    def _handle_or(self, node_inputs, node_outputs, params):
        """Handle JAX or primitive."""
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])
        node = helper.make_node(
            "Or",
            inputs=input_names,
            outputs=[output_name],
            name=self._get_unique_name("or")
        )
        self.nodes.append(node)

    def _handle_xor(self, node_inputs, node_outputs, params):
        """Handle JAX xor primitive."""
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])
        node = helper.make_node(
            "Xor",
            inputs=input_names,
            outputs=[output_name],
            name=self._get_unique_name("xor")
        )
        self.nodes.append(node)


    def _handle_lt(self, node_inputs, node_outputs, params):
        """Handle JAX lt primitive"""
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])
        node = helper.make_node(
            "Less",
            inputs = input_names,
            outputs=[output_name],
            name=self._get_unique_name("less")
        )
        self.nodes.append(node)

    def _handle_gt(self, node_inputs, node_outputs, params):
        """Handle JAX gt primitive"""
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])
        node = helper.make_node(
            "Greater",
            inputs = input_names,
            outputs=[output_name],
            name=self._get_unique_name("greater")
        )
        self.nodes.append(node)
    
    def _handle_max(self, node_inputs, node_outputs, params):
        """Handle JAX max primitive."""
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])

        node = helper.make_node(
            "Max",
            inputs=input_names,
            outputs=[output_name],
            name=self._get_unique_name("max")
        )
        self.nodes.append(node)

    def _handle_min(self, node_inputs, node_outputs, params):
        """Handle JAX min primitive."""
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])

        node = helper.make_node(
            "Min",
            inputs=input_names,
            outputs=[output_name],
            name=self._get_unique_name("min")
        )
        self.nodes.append(node)

    def _handle_select_n(self, node_inputs, node_outputs, params):
        """Handle JAX select_n primitive."""
        condition_name = self._get_name(node_inputs[0])
        false_name = self._get_name(node_inputs[1])
        true_name = self._get_name(node_inputs[2])
        output_name = self._get_var_name(node_outputs[0])
        node = helper.make_node(
            "Where",
            inputs=[condition_name, true_name, false_name],
            outputs=[output_name],
            name=self._get_unique_name("where")
        )
        self.nodes.append(node)

    
    def _handle_dot_general(self, node_inputs, node_outputs, params):
        """Handle JAX dot_general primitive."""
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])
        
        # Extract dot_general parameters
        dimension_numbers, precision = params["dimension_numbers"], params.get("precision", None)
        ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers
        
        # For simple case where it's a standard matrix multiplication
        if len(lhs_contract) == 1 and len(rhs_contract) == 1 and len(lhs_batch) == 0:
            node = helper.make_node(
                "MatMul",
                inputs=input_names,
                outputs=[output_name],
                name=self._get_unique_name("matmul")
            )
            self.nodes.append(node)
        else:
            # For more complex cases, we would need to add transposes and reshapes
            # This is a simplified implementation for the common case
            raise NotImplementedError(
                f"Complex dot_general not yet implemented: {dimension_numbers}"
            )
    
    def _handle_reduce_sum(self, node_inputs, node_outputs, params):
        """Handle JAX reduce_sum primitive."""
        input_name = self._get_name(node_inputs[0])
        output_name = self._get_var_name(node_outputs[0])
        
        # Get axes and create constant for it
        axes = params["axes"]
        axes_name = self._get_constant_name(np.array(axes, dtype=np.int64))
        
        # Create ReduceSum node
        node = helper.make_node(
            "ReduceSum",
            inputs=[input_name, axes_name],
            outputs=[output_name],
            name=self._get_unique_name("reduce_sum"),
            keepdims=0 if not params.get("keepdims", False) else 1
        )
        self.nodes.append(node)
    
    def _handle_reduce_max(self, node_inputs, node_outputs, params):
        """Handle JAX reduce_max primitive."""
        input_name = self._get_name(node_inputs[0])
        output_name = self._get_var_name(node_outputs[0])
        
        # Get axes and create constant for it
        axes = params["axes"]
        axes_name = self._get_constant_name(np.array(axes, dtype=np.int64))
        
        # Create ReduceMax node
        node = helper.make_node(
            "ReduceMax",
            inputs=[input_name, axes_name],
            outputs=[output_name],
            name=self._get_unique_name("reduce_max"),
            keepdims=0 if not params.get("keepdims", False) else 1
        )
        self.nodes.append(node)
    
    def _handle_reduce_min(self, node_inputs, node_outputs, params):
        """Handle JAX reduce_min primitive."""
        input_name = self._get_name(node_inputs[0])
        output_name = self._get_var_name(node_outputs[0])
        
        # Get axes and create constant for it
        axes = params["axes"]
        axes_name = self._get_constant_name(np.array(axes, dtype=np.int64))
        
        # Create ReduceMin node
        node = helper.make_node(
            "ReduceMin",
            inputs=[input_name, axes_name],
            outputs=[output_name],
            name=self._get_unique_name("reduce_min"),
            keepdims=0 if not params.get("keepdims", False) else 1
        )
        self.nodes.append(node)
    
    def _handle_gather(self, node_inputs, node_outputs, params):
        input_names = [self._get_name(imp) for imp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])

        node = helper.make_node(
            "GatherElements",
            inputs = input_names,
            outputs=[output_name],
            name=self._get_unique_name("gather"),
        )

        self.nodes.append(node)


    def _handle_scatter_add(self, node_inputs, node_outputs, params):
        input_names = [self._get_name(imp) for imp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])

        node = helper.make_node(
            "ScatterElements",
            inputs = input_names,
            outputs=[output_name],
            name=self._get_unique_name("scatter_add"),
            reduction="add"
        )

        self.nodes.append(node)

    def _handle_argmax(self, node_inputs, node_outputs, params):
        """Handle JAX argmax primitive."""
        input_name = self._get_name(node_inputs[0])
        intermediate_name = self._get_unique_name("argmax_intermediate")
        output_name = self._get_var_name(node_outputs[0])

        axis = params["axes"][0]
        index_dtype = params["index_dtype"]
        keepdims = 1 if "keepdims" in params else 0

        node_1 = helper.make_node(
            "ArgMax",
            inputs = [input_name],
            outputs = [intermediate_name],
            name = self._get_unique_name("argmax"),
            axis=axis,
            keepdims = keepdims
        )
        self.nodes.append(node_1)

        # Slight quirk in ONNX: argmax returns int64, but we can always(?) cast to int32.

        node_2 = helper.make_node(
            "Cast",
            inputs = [intermediate_name],
            outputs  = [output_name],
            to = TensorProto.INT32
        )
        self.nodes.append(node_2)




    def _handle_argmin(self, node_inputs, node_outputs, params):
        """Handle JAX argmin primitive."""
        input_name = self._get_name(node_inputs[0])
        intermediate_name = self._get_unique_name("argmax_intermediate")
        output_name = self._get_var_name(node_outputs[0])

        axis = params["axes"][0]
        index_dtype = params["index_dtype"]
        keepdims = params["keepdims"]
        node_1 = helper.make_node(
            "ArgMin",
            inputs = [input_name],
            outputs = [intermediate_name],
            name = self._get_unique_name("argmin"),
            axis=axis,
            keepdims = keepdims
        )

        self.nodes.append(node_1)

        # Slight quirk in ONNX: argmin returns int64, but we can always(?) cast to int32.

        node_2 = helper.make_node(
            "Cast",
            inputs = [intermediate_name],
            outputs  = [output_name],
            to = TensorProto.INT32
        )
        self.nodes.append(node_2)


    def _handle_square(self, node_inputs, node_outputs, params):
        """Handle JAX square primitive."""
        input_name = self._get_name(node_inputs[0])
        output_name = self._get_var_name(node_outputs[0])

        power_name = self._get_constant_name(np.array(2, dtype=np.int32))

        node = helper.make_node(
            "Pow",
            inputs = [input_name, power_name],
            outputs = [output_name],
            name = self._get_unique_name("square")
        )

        self.nodes.append(node)
    
    def _handle_integer_pow(self, node_inputs, node_outputs, params):
        """Handle JAX integer pow primitive."""
        input_name = self._get_name(node_inputs[0])
        output_name = self._get_var_name(node_outputs[0])

        power_name = self._get_constant_name(np.array(params['y'], dtype=np.int32))

        node = helper.make_node(
            "Pow",
            inputs = [input_name, power_name],
            outputs = [output_name],
            name = self._get_unique_name("square")
        )

        self.nodes.append(node)


    def _handle_sqrt(self, node_inputs, node_outputs, params):
        """Handle JAX sqrt primitive."""
        input_name = self._get_name(node_inputs[0])
        output_name = self._get_var_name(node_outputs[0])
        
        node = helper.make_node(
            "Sqrt",
            inputs=[input_name],
            outputs=[output_name],
            name=self._get_unique_name("sqrt")
        )
        self.nodes.append(node)

    
    def _handle_exp(self, node_inputs, node_outputs, params):
        """Handle JAX exp primitive."""
        input_name = self._get_name(node_inputs[0])
        output_name = self._get_var_name(node_outputs[0])
        
        node = helper.make_node(
            "Exp",
            inputs=[input_name],
            outputs=[output_name],
            name=self._get_unique_name("exp")
        )
        self.nodes.append(node)
    
    def _handle_log(self, node_inputs, node_outputs, params):
        """Handle JAX log primitive."""
        input_name = self._get_name(node_inputs[0])
        output_name = self._get_var_name(node_outputs[0])
        
        node = helper.make_node(
            "Log",
            inputs=[input_name],
            outputs=[output_name],
            name=self._get_unique_name("log")
        )
        self.nodes.append(node)
    
    def _handle_tanh(self, node_inputs, node_outputs, params):
        """Handle JAX tanh primitive."""
        input_name = self._get_name(node_inputs[0])
        output_name = self._get_var_name(node_outputs[0])
        
        node = helper.make_node(
            "Tanh",
            inputs=[input_name],
            outputs=[output_name],
            name=self._get_unique_name("tanh")
        )
        self.nodes.append(node)
    
    def _handle_sigmoid(self, node_inputs, node_outputs, params):
        """Handle JAX sigmoid primitive."""
        input_name = self._get_name(node_inputs[0])
        output_name = self._get_var_name(node_outputs[0])
        
        node = helper.make_node(
            "Sigmoid",
            inputs=[input_name],
            outputs=[output_name],
            name=self._get_unique_name("sigmoid")
        )
        self.nodes.append(node)
    
    def _handle_iota(self, node_inputs, node_outputs, params):
        """Handle JAX iota primitive."""
        # input_name = self._get_name(node_inputs[0])
        output_name = self._get_var_name(node_outputs[0])

        dtype = params["dtype"] # TODO: Use dtype
        if dtype != jnp.int32:
            raise NotImplementedError("dtype not implemented")
        shape = params["shape"]

        L = shape[0] # TODO: consider when len(shape) > 1
        start_name = self._get_constant_name(np.array(0, dtype=np.int32))
        end_name = self._get_constant_name(np.array(L, dtype=np.int32))
        step_name = self._get_constant_name(np.array(1, dtype=np.int32))
        
        node = helper.make_node(
            "Range",
            inputs=[start_name, end_name, step_name],
            outputs=[output_name],
            name=self._get_unique_name("iota")
        )
        self.nodes.append(node)
    
    def _handle_reshape(self, node_inputs, node_outputs, params):
        """Handle JAX reshape primitive."""
        input_name = self._get_name(node_inputs[0])
        output_name = self._get_var_name(node_outputs[0])
        
        # Get new shape and create constant for it
        new_shape = params["new_sizes"]
        shape_name = self._get_constant_name(np.array(new_shape, dtype=np.int64))
        
        # Create Reshape node
        node = helper.make_node(
            "Reshape",
            inputs=[input_name, shape_name],
            outputs=[output_name],
            name=self._get_unique_name("reshape")
        )
        self.nodes.append(node)
    
    def _handle_transpose(self, node_inputs, node_outputs, params):
        """Handle JAX transpose primitive."""
        input_name = self._get_name(node_inputs[0])
        output_name = self._get_var_name(node_outputs[0])
        
        # Get permutation
        permutation = params["permutation"]
        
        # Create Transpose node
        node = helper.make_node(
            "Transpose",
            inputs=[input_name],
            outputs=[output_name],
            name=self._get_unique_name("transpose"),
            perm=permutation
        )
        self.nodes.append(node)
    
    def _handle_squeeze(self, node_inputs, node_outputs, params):
        """Handle JAX squeeze primitive."""
        input_name = self._get_name(node_inputs[0])
        output_name = self._get_var_name(node_outputs[0])
        
        # Get permutation
        dims = params["dimensions"]
        axes = self._get_constant_name(np.array(dims, dtype=np.int64))
        
        # Create Transpose node
        node = helper.make_node(
            "Squeeze",
            inputs=[input_name, axes],
            outputs=[output_name],
            name=self._get_unique_name("squeeze"),
        )
        self.nodes.append(node)

    
    def _handle_broadcast_in_dim(self, node_inputs, node_outputs, params):
        """Handle JAX broadcast_in_dim primitive."""
        input_name = self._get_name(node_inputs[0])
        output_name = self._get_var_name(node_outputs[0])
        
        # Get broadcast dimensions and shape
        broadcast_dimensions = params["broadcast_dimensions"]
        shape = params["shape"]
        
        # Create constants for shape
        shape_name = self._get_constant_name(np.array(shape, dtype=np.int64))
        
        # ONNX doesn't have a direct equivalent to broadcast_in_dim
        # We'll use a combination of Reshape and Expand
        
        # First reshape to add singleton dimensions
        reshape_output = self._get_unique_name("reshape_output")
        reshape_shape = []
        idx = 0
        for i in range(len(shape)):
            if i in broadcast_dimensions:
                reshape_shape.append(1 if idx >= len(node_inputs[0].aval.shape) else node_inputs[0].aval.shape[idx])
                idx += 1
            else:
                reshape_shape.append(1)
        
        reshape_shape_name = self._get_constant_name(np.array(reshape_shape, dtype=np.int64))
        
        reshape_node = helper.make_node(
            "Reshape",
            inputs=[input_name, reshape_shape_name],
            outputs=[reshape_output],
            name=self._get_unique_name("reshape_for_broadcast")
        )
        self.nodes.append(reshape_node)
        
        # Then expand to target shape
        expand_node = helper.make_node(
            "Expand",
            inputs=[reshape_output, shape_name],
            outputs=[output_name],
            name=self._get_unique_name("expand")
        )
        self.nodes.append(expand_node)
    
    def _handle_slice(self, node_inputs, node_outputs, params):
        """Handle JAX slice primitive."""
        input_name = self._get_name(node_inputs[0])
        output_name = self._get_var_name(node_outputs[0])
        
        # Get slice parameters
        start_indices = params["start_indices"]
        starts_name = self._get_constant_name(np.array(start_indices, dtype=np.int64))
        limit_indices = params["limit_indices"]
        ends_name = self._get_constant_name(np.array(limit_indices, dtype=np.int64))
        axes_name = self._get_constant_name(np.array(list(range(len(start_indices))), dtype=np.int64))
        inputs = [input_name, starts_name, ends_name, axes_name]

        if "strides" in params and params["strides"]:
            # Create constants for ONNX Slice op
            # strides = params["strides"] if "strides" in params else [1] * len(start_indices)
            strides = params["strides"] 
            steps_name = self._get_constant_name(np.array(strides, dtype=np.int64))
            inputs.append(steps_name)
        
        
        # Create Slice node
        node = helper.make_node(
            "Slice",
            inputs=inputs,
            outputs=[output_name],
            name=self._get_unique_name("slice")
        )
        self.nodes.append(node)
    
    def _handle_concatenate(self, node_inputs, node_outputs, params):
        """Handle JAX concatenate primitive."""
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])
        
        # Get concatenation axis
        dimension = params["dimension"]
        
        # Create Concat node
        node = helper.make_node(
            "Concat",
            inputs=input_names,
            outputs=[output_name],
            name=self._get_unique_name("concat"),
            axis=dimension
        )
        self.nodes.append(node)
    
    def _handle_conv(self, node_inputs, node_outputs, params):
        """Handle JAX conv_general_dilated primitive."""
        # This is a simplified implementation for common cases
        input_name = self._get_name(node_inputs[0])  # input
        filter_name = self._get_name(node_inputs[1])  # weights
        output_name = self._get_name(node_outputs[0])
        
        # Extract parameters
        dimension_numbers = params["dimension_numbers"]
        window_strides = params["window_strides"]
        padding = params["padding"]
        lhs_dilation = params.get("lhs_dilation", (1,) * (len(window_strides)))
        rhs_dilation = params.get("rhs_dilation", (1,) * (len(window_strides)))
        
        # Parse dimension numbers
        lhs_spec, rhs_spec, out_spec = dimension_numbers
        
        # ONNX Conv expects specific dimension ordering
        # N=batch, C=channel, D/H/W=spatial dims
        
        # Simple case: assume standard dimension ordering
        # This is highly simplified and won't work for all JAX conv cases
        node = helper.make_node(
            "Conv",
            inputs=[input_name, filter_name],
            outputs=[output_name],
            name=self._get_unique_name("conv"),
            kernel_shape=node_inputs[1].aval.shape[2:],
            strides=window_strides,
            dilations=rhs_dilation,
            pads=sum(padding, ())  # Flatten [(p0, p0), (p1, p1), ...] to [p0, p0, p1, p1, ...]
        )
        self.nodes.append(node)
    
    def _handle_max_pool(self, node_inputs, node_outputs, params):
        """Handle JAX max_pool primitive."""
        input_name = self._get_name(node_inputs[0])
        output_name = self._get_var_name(node_outputs[0])
        
        # Extract parameters
        window_dimensions = params["window_dimensions"]
        window_strides = params["window_strides"]
        padding = params["padding"]
        
        # Create MaxPool node
        node = helper.make_node(
            "MaxPool",
            inputs=[input_name],
            outputs=[output_name],
            name=self._get_unique_name("maxpool"),
            kernel_shape=window_dimensions[2:],
            strides=window_strides,
            pads=sum(padding, ())  # Flatten padding
        )
        self.nodes.append(node)
    
    def _handle_avg_pool(self, node_inputs, node_outputs, params):
        """Handle JAX avg_pool primitive."""
        input_name = self._get_name(node_inputs[0])
        output_name = self._get_var_name(node_outputs[0])
        
        # Extract parameters
        window_dimensions = params["window_dimensions"]
        window_strides = params["window_strides"]
        padding = params["padding"]
        
        # Create AveragePool node
        node = helper.make_node(
            "AveragePool",
            inputs=[input_name],
            outputs=[output_name],
            name=self._get_unique_name("avgpool"),
            kernel_shape=window_dimensions[2:],
            strides=window_strides,
            pads=sum(padding, ())  # Flatten padding
        )
        self.nodes.append(node)

    def _handle_sort(self, node_inputs, node_outputs, params):
        """Handle JAX sort primitive"""
        input_name = self._get_name(node_inputs[0]) 
        shape_name = self._get_unique_name("sort_shape")
        value_name = self._get_var_name(node_outputs[0])
        indices_name = self._get_unique_name("sort_indices_output")

        if "axis" in params:
            axis = params["axis"]
            K = node_inputs[0].aval.shape[axis]
            raise NotImplementedError("sort axis not supported yet")
        else:
            node = helper.make_node(
                "Shape",
                inputs=[input_name],
                outputs=[shape_name],
                name = self._get_unique_name("shape")
            )
            self.nodes.append(node)
        
        # to make sort more generic, we first find the shape
        node = helper.make_node(
            "TopK",
            inputs = [input_name, shape_name],
            outputs=[value_name, indices_name],
            name=self._get_unique_name("sort"),
            largest = 0
        )

        self.nodes.append(node)

    def _handle_random_uniform(self, node_inputs, node_outputs, params):
        # input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])
        shape = node_outputs[0].aval.shape
        if shape == ():
            shape = (1,)
        node = helper.make_node(
            "RandomUniform",
            inputs = [],
            outputs = [output_name],
            name=self._get_unique_name("random_uniform"),
            shape=shape
        )
        self.nodes.append(node)

    def _handle_random_normal(self, node_inputs, node_outputs, params):
        # input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])
        shape = node_outputs[0].aval.shape
        if shape == ():
            shape = (1,)
        node = helper.make_node(
            "RandomNormal",
            inputs = [],
            outputs = [output_name],
            name=self._get_unique_name("random_normal"),
            shape=shape
        )
        self.nodes.append(node)
    
    def _handle_random_gamma(self, node_inputs, node_outputs, params):
        """
        Handle JAX gamma primitive

        between Marsaglia-Tang and Cheng, we decided on the former due to the low rejection rate
        https://kth.diva-portal.org/smash/get/diva2:1935824/FULLTEXT02.pdf

        d = α - 1/3
        c = 1/sqrt(9d)

        repeat
            sample Z ~ Normal(0,1)
            V = (1 + cZ)^3
            sample U ~ Uniform(0,1)
            X = dV
            if V > 0 and log(U) < 1/2 Z^2 + d - dV + dlog(V) then
                accept X
            endif

        until X is accepted
        return X
        """
        # Create a jaxpr and run JaxprToOnnx to build the CG

        shape = node_inputs[1].aval.shape
        key = jax.random.key(0)
        alpha = jnp.zeros(shape)

        # TODO: Case 0 < alpha <= 1/3 not handled
        subconverter = JaxprToOnnx(self.name_counter + 1)
        if "log_space" in params and params["log_space"]:
            subconverter.trace_jaxpr(gamma_log, (key, alpha))
        else:
            subconverter.trace_jaxpr(gamma, (key, alpha))

        # connect inputs/outputs to outer jaxpr
        nodes = subconverter.nodes
        initializers = subconverter.initializers
        inputs = subconverter.inputs
        outputs = subconverter.outputs

        assert len(node_inputs) == len(inputs)
        assert len(node_outputs) == len(outputs)


        for o_invar, i_invar in zip(node_inputs, inputs):
            o_invar_name = self._get_name(o_invar)
            i_invar_name = i_invar.name
            node = helper.make_node(
                "Identity",
                inputs = [o_invar_name],
                outputs = [i_invar_name],
                name = self._get_unique_name("gamma_input")
            )
            self.nodes.append(node)

        self.nodes += nodes
        self.initializers += initializers
        self.name_counter += (subconverter.name_counter - subconverter._name_counter_init)

        for o_outvar, i_outvar in zip(node_outputs, outputs):
            o_outvar_name = self._get_name(o_outvar)
            i_outvar_name = i_outvar.name
            node = helper.make_node(
                "Identity",
                inputs = [i_outvar_name],
                outputs = [o_outvar_name],
                name = self._get_unique_name("gamma_output")
            )
            self.nodes.append(node)

    def _handle_convert_element_type(self, node_inputs, node_outputs, params):
        input_names = [self._get_name(inp) for inp in node_inputs]
        output_name = self._get_var_name(node_outputs[0])   

        new_dtype = self._numpy_dtype_to_onnx(params["new_dtype"])
        node = helper.make_node(
            "Cast",
            inputs = input_names,
            outputs = [output_name],
            name = self._get_unique_name("convert_element_type"),
            to = new_dtype
        )
        self.nodes.append(node)

    def _handle_device_put(self, node_inputs, node_outputs, params):
        name = self._get_unique_name("const")
        # Convert to numpy and create tensor
        val = node_inputs[0]
        actual_val = val.val
        
        np_val = np.array(actual_val)
        if np_val.dtype == np.int64:
            np_val = np_val.astype(np.int32)
        elif np_val.dtype == np.float64:
            np_val = np_val.astype(np.float32)

        tensor = helper.make_tensor(
            name=name,
            data_type=self._numpy_dtype_to_onnx(np_val.dtype),
            dims=np_val.shape,
            vals=np_val.flatten().tolist(),
        )
        self.initializers.append(tensor)
        # return name
        input_names = [name]
        output_name = self._get_var_name(node_outputs[0])

        node = helper.make_node(
            "Identity",
            inputs = input_names,
            outputs = [output_name],
            name = self._get_unique_name("device_put")
        )
        self.nodes.append(node)
    
    
    def _process_pjit(self, jaxpr):
        closed_jaxpr = jaxpr.params["jaxpr"]
        if not isinstance(closed_jaxpr, jax._src.core.ClosedJaxpr):
            raise ValueError("Expected ClosedJaxpr in pjit.param[jaxpr]")
        
        name = jaxpr.params["name"]
        if name == "_normal":
            self._handle_random_normal(jaxpr.invars, jaxpr.outvars, jaxpr.params)
        elif name == "_uniform":
            self._handle_random_uniform(jaxpr.invars, jaxpr.outvars, jaxpr.params)
        elif name == "_gamma":
            self._process_closed_jaxpr(jaxpr)
        elif name == "clip":
            self._process_closed_jaxpr(jaxpr)
        elif name == "sort":
            self._process_closed_jaxpr(jaxpr)
        elif name == "_where":
            self._process_closed_jaxpr(jaxpr)
        elif name == "_gumbel":
            self._process_closed_jaxpr(jaxpr)
        elif name == "_dirichlet":
            self._process_closed_jaxpr(jaxpr)
        else:
            raise NotImplementedError(f"pjit {jaxpr.params["name"]} not yet handled")

    def _process_eqn(self, jaxpr):
        """Process a single JAXPR equation."""
        if hasattr(jaxpr, "primitive"):
            primitive = jaxpr.primitive
            if primitive.name == "pjit":
                self._process_pjit(jaxpr)
            elif primitive in self.primitive_handlers:
                self.primitive_handlers[primitive](
                    jaxpr.invars, jaxpr.outvars, jaxpr.params
                )
            else:
                raise NotImplementedError(f"Primitive {primitive} not implemented")
        else:
            # Handle call primitives or other special cases
            raise NotImplementedError(f"Non-primitive equation: {jaxpr}")

    def _process_closed_jaxpr(self, jaxpr): 
        # TODO: CONFUSING, `jaxpr` is a JaxprEqn which contains the ClosedJaxpr
        assert isinstance(jaxpr, jax._src.core.JaxprEqn)

        closed_jaxpr = jaxpr.params["jaxpr"]
        node_inputs = jaxpr.invars
        node_outputs = jaxpr.outvars

        subconverter = JaxprToOnnx(self.name_counter + 1)
        subconverter._process_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts)


        nodes = subconverter.nodes
        initializers = subconverter.initializers
        inputs = subconverter.inputs
        outputs = subconverter.outputs

        assert len(node_inputs) == len(inputs)
        assert len(node_outputs) == len(outputs)


        for o_invar, i_invar in zip(node_inputs, inputs):
            o_invar_name = self._get_name(o_invar)
            i_invar_name = i_invar.name
            node = helper.make_node(
                "Identity",
                inputs = [o_invar_name],
                outputs = [i_invar_name],
                name = self._get_unique_name("pjit_input")
            )
            self.nodes.append(node)

        self.nodes += nodes
        self.initializers += initializers
        self.name_counter += (subconverter.name_counter - subconverter._name_counter_init)

        for o_outvar, i_outvar in zip(node_outputs, outputs):
            o_outvar_name = self._get_name(o_outvar)
            i_outvar_name = i_outvar.name
            node = helper.make_node(
                "Identity",
                inputs = [i_outvar_name],
                outputs = [o_outvar_name],
                name = self._get_unique_name("pjit_output")
            )
            self.nodes.append(node)
    
    def _process_jaxpr(self, jaxpr, consts):
        """Process a JAXPR and convert it to ONNX nodes."""
        # Setup inputs
        for var in jaxpr.invars:
            self._add_input(var, var.aval.shape, var.aval.dtype)
        
        # Setup constants
        for i, const in enumerate(consts):
            const_name = self._get_constant_name(const)
            const_var = jaxpr.constvars[i]
            self.var_to_name[const_var] = const_name
        
        # Process all equations in the JAXPR
        for eqn in jaxpr.eqns:
            self._process_eqn(eqn)
        
        # Setup outputs
        for var in jaxpr.outvars:
            self._add_output(var, var.aval.shape, var.aval.dtype)
    
    def trace_jaxpr(self, fn, example_args):
        # Reset state
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.initializers = []
        self.value_info = []
        self.name_counter = self._name_counter_init
        self.var_to_name = {}

        # Get JAXPR from the function
        closed_jaxpr = jax.make_jaxpr(fn)(*example_args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.consts
        
        self._process_jaxpr(jaxpr, consts)

    def convert(self, fn, example_args, output_path='model.onnx', model_name = "jax_model"):
        """
        Convert a JAX function to ONNX.
        
        Args:
            fn: JAX function to convert
            example_args: Example input arguments to trace the function
            output_path: Path to save the ONNX model
            
        Returns:
            Path to the saved ONNX model
        """
        self.trace_jaxpr(fn, example_args)
                
        graph = helper.make_graph(
            nodes=self.nodes,
            name=model_name,
            inputs=self.inputs,
            outputs=self.outputs,
            initializer=self.initializers,
            value_info=self.value_info
        )
        
        # Create ONNX model
        onnx_model = helper.make_model(
            graph,
            producer_name="jaxpr_to_onnx",
            opset_imports=[helper.make_opsetid("", 21)]
        )
        
        # Save model
        onnx.save(onnx_model, output_path)
        return output_path


# Example usage
def example():
    # Define a simple JAX function
    def example_fn(x, y):
        return jnp.sum(x * y + 1.0, axis=0)
        # return x * y + 1.0
    
    # Create inputs
    x = jnp.ones((3, 4))
    y = jnp.ones((3, 4)) * 2.0
    
    # Convert to ONNX
    converter = JaxprToOnnx()
    model_path = converter.convert(example_fn, (x, y), "example_model.onnx")
    print(f"ONNX model saved to: {model_path}")

def gamma(key, alpha):
    d = alpha - 1/3
    c = 1 / jnp.sqrt(9 * d)
    z = jax.random.normal(key, alpha.shape)
    v = (1+c*z)**3
    u = jax.random.uniform(key, alpha.shape)
    x = d * v

    acceptance = (v > 0) & (jnp.log(u) < (0.5 * z**2 + d - d*v + d * jnp.log(v)))

    z = jax.random.normal(key, alpha.shape)
    v = (1+c*z)**3
    x = jnp.where(acceptance, x, d * v)

    # clip when alpha = 0
    x = jnp.where(alpha == 0, 0.0, x)

    return x

def gamma_log(key, alpha):
    x = gamma(key, alpha)
    return jnp.log(x)

# More complex example
# def complex_example():
    # Define a small CNN
#     def cnn_fn(x):
#         # Convolution
#         w = jnp.ones((16, 3, 3, 3))  # [out_channels, in_channels, height, width]
#         x = jax.lax.conv_general_dilated(
#             x, w, 
#             window_strides=(1, 1),
#             padding="SAME",
#             dimension_numbers=('NHWC', 'OHWI', 'NHWC')
#         )
#         # ReLU
#         x = jnp.maximum(x, 0)
#         # Max pooling
#         # x = jax.lax.reduce_window(
#         #     x,
#         #     init_value=-jnp.inf,
#         #     computation=jax.lax.max,
#         #     window_dimensions=(1, 2, 2, 1),
#         #     window_strides=(1, 2, 2, 1),
#         #     padding='VALID'
#         # )
#         # Flatten
#         batch_size = x.shape[0]
#         x = x.reshape(batch_size, -1)
#         # Dense layer
#         w2 = jnp.ones((x.shape[1], 10))
#         x = jnp.dot(x, w2)
#         # Softmax
#         x = jax.nn.softmax(x)
#         return x
    
#     # Create input
#     x = jnp.ones((1, 28, 28, 3))  # [batch, height, width, channels]
    
#     # Convert to ONNX
#     converter = JaxprToOnnx()
#     model_path = converter.convert(cnn_fn, (x,), "cnn_model.onnx")
#     print(f"CNN model saved to: {model_path}")

# def random_example():
#     def model(key):
#         x = jax.random.normal(key)
#         y = x + 1.0
#         return y
    
#     key = jax.random.key(0)
#     converter = JaxprToOnnx()
#     model_path = converter.convert(model, (key,), "gaussian.onnx")
#     print(f"Gaussian noise saved to {model_path}")
# if __name__ == "__main__":
#     # example()
#     # Uncomment to run the more complex example
#     # complex_example()
#     random_example()