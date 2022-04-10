import dezero.layers as L
import dezero.utils as utils
import dezero.functions as F

class Model(L.Layer):#Layer에서 plot만 추가
    def plot(self, *inputs, to_file='model.png'):
        y=self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):#2nd argument는 layer크기들을 의미.
        super().__init__()
        self.activation=activation
        self.layers=[]

        for i, out_size in enumerate(fc_output_sizes):
            layer=L.Linear(out_size)
            setattr(self, 'l'+str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x=self.activation(l(x))
        return self.layers[-1](x)
#model=MLP((10,1))->2층 MLP
#model=MLP((10,20,30,40,1))->5층 MLP
