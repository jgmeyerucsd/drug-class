from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.io import output_notebook

source = ColumnDataSource(data=dict(
    x=PC1s,
    y=PC2s,
    desc=sample_names,
    colors=colors
))

TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
    ("desc", "@desc"),
]

output_notebook()

p = figure(tooltips=TOOLTIPS, plot_width=800, plot_height=600)
p.xaxis.axis_label = 'PC1 ({})'.format(round(pca.explained_variance_ratio_[0],2))
p.yaxis.axis_label = 'PC2 ({})'.format(round(pca.explained_variance_ratio_[1],2))

p.circle('x','y', color='colors', fill_alpha=0.2, size=20, source=source, legend='colors')

p.yaxis.axis_label_text_font_size = "25pt"
p.yaxis.major_label_text_font_size = "20pt"
p.yaxis.major_label_text_font = "oswald"
p.yaxis.major_label_text_color = "black"
p.yaxis.axis_label_text_font = "oswald"
p.yaxis.axis_label_text_color = "black"

p.xaxis.axis_label_text_font_size = "25pt"
p.xaxis.major_label_text_font_size = "20pt"
p.yaxis.axis_label_text_color = "black"
p.xaxis.major_label_text_font = "oswald"
p.xaxis.axis_label_text_font = "oswald"
p.xaxis.axis_label_text_color = "black"

p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

p.xgrid.visible = False
p.ygrid.visible = False

p.legend.location = "top_right"
p.legend.border_line_color = None
p.outline_line_color = None

p.output_backend = "svg"

show(p)
