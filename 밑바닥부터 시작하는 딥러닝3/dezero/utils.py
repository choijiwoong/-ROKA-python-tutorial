import os
import subprocess
import dezero.cuda as cuda

#[가시화툴]
def _dot_var(v, verbose=False):#Variable전용 dot
    dot_var='{} [label="{}", color=orange, style=filled]\n'

    name='' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name+=': '
        name+=str(v.shape)+' '+str(v.dtype)
    return dot_var.format(id(v), name)#주소, 이름, 형태, 데이터타입정보 담긴 dot

def _dot_func(f):#Function전용 dot(input과 output의 관계 형성)
    dot_func='{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt=dot_func.format(id(f), f.__class__.__name__)

    dot_edge='{}->{}\n'#함수의 input, output의 관계 표시
    for x in f.inputs:
        txt+=dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt+=dot_edge.format(id(f), id(y()))#y는 약한참조
    return txt

def get_dot_graph(output, verbose=True):
    txt=''
    funcs=[]
    seen_set=set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)#sort by generation이 없는 이유는 단순히 그래프화이기에 역전파?방향 순서는 상관이 없음
            seen_set.add(f)
    add_func(output.creator)
    
    txt+=_dot_var(output,verbose)
    while funcs:
        func=funcs.pop()
        txt+=_dot_func(func)#함수++
        for x in func.inputs:
            txt+=_dot_var(x, verbose)#입력x++

            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g {\n'+txt+'}'#Dot문법에 따라 dot리턴.

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph=get_dot_graph(output, verbose)

    tmp_dir=os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path=os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    extension=os.path.splitext(to_file)[1][1:]#확장자
    cmd='dot {} -T {} -o {}'.format(graph_path, extension, to_file)#Dot파일을 png혹은 pdf형식에 따라 ~로 저장
    subprocess.run(cmd, shell=True)

#툴
def reshape_sum_backward(gy, x_shape, axis, keepdims):#reshape할때 sum의 axis, keepdims인자 고려
    ndim=len(x_shape)
    tupled_axis=axis
    if axis is None:
        tupled_axis=None
    elif not isinstance(axis, tuple):
        tupled_acis=(axis,)

    if not (ndim==0 or tupled_axis is None or keepdims):#튜플 혹은 ndim이나 keepdims가 존재
        actual_axis=[a if a>=0 else a+ndim for a in tupled_axis]#-값도 고려하여 실제 축값으로 변경
        shape=list(gy.shape)#순전파 출력의 크기(현재 입력)
        for a in sorted(actual_axis):
            shape.insert(a,1)#실제 축값에 1을 삽입
    else:#스칼라 혹은 기타인자 None
        shape=gy.shape

    gy=gy.reshape(shape)#gy's shape, tupled_axis, keepdims에 따라 성형만 해준다.
    return gy

def sum_to(x, shape):#broadcast된 (2,1)->(2,4)를 다시 (2,1)로(lead는 뭔지 모르겠음..ㅠ대충 아예 (2,)->(2,4)이런경우 말하는거같긴함..)
    ndim=len(shape)
    lead=x.ndim-ndim#얼마나 더해야하는지
    lead_axis=tuple(range(lead))

    axis=tuple([i+lead for i, sx in enumerate(shape) if sx==1])#현재 1인 후보값들
    y=x.sum(lead_axis+axis, keepdims=True)#x.shape와 shape의 차이나는 부분을 매꿀
    if lead>0:
        y=y.squeeze(lead_axis)
    return y

def logsumexp(x, axis=1):
    xp=cuda.get_array_module(x)
    m=x.max(axis=axis, keepdims=True)
    y=x-m#저..정규화?
    xp.exp(y, out=y)#y=e^y
    s=y.sum(axis=axis, keepdims=True)#sum
    xp.log(s, out=s)#log
    m+=s#이게 쉬발 뭐하는 함수고? 그냥 softmax_simple쓰련다
    return m

#[아몰랑]
import os
import urllib.request
import subprocess

cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')

def get_file(url, file_name=None):
    """Download a file from the `url` if it is not in the cache.

    The file at the `url` is downloaded to the `~/.dezero`.

    Args:
        url (str): URL of the file.
        file_name (str): Name of the file. It `None` is specified the original
            file name is used.

    Returns:
        str: Absolute path to the saved file.
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path

def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError

def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0: p = 100.0
    if i >= 30: i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')
