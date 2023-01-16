import math

class Value:
    def __init__(self, data, children=(), op=None):
        self.data = float(data)
        self.children = children
        self.grad = 0
        self.op = op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data},grad={self.grad},op={self.op})"

    '''
        The chain rule is core to performing the backwards pass
        
                                           +-----+       +-----+
                                           |  a  |       |  b  |
                                           +-----+       +-----+
                                              \             /
                                               \           /
                                                +----------+
                                                |  (plus)  |
                                                |    c     |
                                                +----------+
                                                     | 
                                                     |   +-----+
                                                     |   |  d  |
                                                     |   +-----+ 
                                                     |      /
                                                     |     /
                                                +---------+
                                                |  (mul)  |
                                                |    L    |
                                                +---------+
        
        We want to compute the partial derivatives of all variables with respect to L so we know how to perturb
        these variables to minimize L. From observation we know that dL/dd = c and dL/dc = d. If we want to compute
        dL/da, we can first write c(a) = a + b. Now because we know dL/dc and c(a) is a function of `a`, the chain rule
        can be applied. dL/da = dL/dc * dc/da = d * 1 = d. So autograd is just using the chain rule + recursion to
        calculate the gradient of each variable. In general you can think of backprop as multiplying the accumulated
        gradient flowing backwards (self.grad) by the local gradient of the current operator and this is all thanks
        to the chain rule. If a variable is used more than once, the gradient passed backward should be added to the existing
        gradient to accurately reflect that variable's total contribution to the loss
    '''

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, [self, other], '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, [self, other], '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward  = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "k must either be int or float"
        out = Value(self.data ** k, [self], '**')
        def _backward():
            self.grad += k * (self.data ** (k - 1)) * out.grad # k * x^(k-1) * g
        out._backward = _backward
        return out

    def tanh(self):
        exp = math.exp(2*self.data)
        out = Value((exp-1)/(exp+1), [self], 'tanh')
        def _backward():
            self.grad += 1 - (out.data ** 2)
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(self.data, 0), [self], 'relu')
        def _backward():
            self.grad += out.grad if self.data > 0 else 0
        out._backward = _backward
        return out


    def backward(self):
        self.grad = 1 # Derivative with respect to yourself is always 1
        self._backward()
        children = self.children
        while len(children) > 0:
            child = children.pop(0)
            child._backward()
            if child.children:
                children.extend(child.children)


    # Rewrite operations below to use primitives implemented above
    def __sub__(self, other):
        return self + -other

    def __neg__(self):
        return -1 * self

    def __rmul__(self, other):
        # called when Python notices the RHS is a Value
        return self * other

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self - other