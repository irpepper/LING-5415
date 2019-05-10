
# coding: utf-8

# In[38]:


from bokeh.plotting import figure, output_file, show
import os
PATH = "outputs\size\\"


# In[39]:


x = []
y = []
for fp in os.listdir(PATH):
    size = float(fp[-3:])
    with open(PATH+fp,"r") as f:
        val = float(f.read()[-6:])
    x.append(size)
    y.append(val)


# In[40]:


# output to static HTML file
output_file("size_evaluation.html")

# create a new plot with a title and axis labels
p = figure(title="Training Set Size vs F1", x_axis_label='% Train', y_axis_label='F1')

# add a line renderer with legend and line thickness
p.line(x, y, legend="F1", line_width=4)
p.circle(x, y, fill_color="blue", line_color="blue", size=6)
p.legend.location = "top_left"
p.xaxis.ticker = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# show the results
show(p)

