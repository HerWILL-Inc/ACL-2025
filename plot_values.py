import plotly.graph_objects as go

fig = go.Figure()

#shape
# x_corr = [0.8883957266807556, 0.6563590168952942, 0.8789744973182678, 0.9452601671218872, 0.6063305139541626, 0.7290314435958862, 0.5224672555923462, 0.9362486004829407, 0.5590897798538208, 0.641208291053772]
# y_corr = [0.8895320892333984, 0.656102180480957, 0.874659538269043, 0.9452641606330872, 0.595225989818573, 0.7241197824478149, 0.5123922228813171, 0.9386897087097168, 0.5862528681755066, 0.6415683031082153]

#stem
x_corr = [0.8586,0.586,0.8643,0.9422,0.5838,0.7008,0.5034,0.9161,0.5416,0.5903]
y_corr = [0.8607,0.5875,0.859,0.9426,0.5789,0.6941,0.4962,0.9189,0.5896,0.5979]

modelnames = ['csebuetnlp/banglabert', 'saiful9379/Bangla_GPT2', 'flax-community/gpt2-bengali', 'ritog/bangla-gpt2', 'csebuetnlp/banglat5', 'neuropark/sahajBERT', 'Kowsher/bangla-bert', 'csebuetnlp/banglishbert', 'sagorsarker/bangla-bert-base', 'shahidul034/text_generation_bangla_model']

fig.add_trace(go.Scatter(
    x=x_corr,
    y=y_corr,
    mode="markers+text",
    textposition="bottom center",
))

fig.update_layout(annotations=[
            go.layout.Annotation(x=x_corr[i],
            y=y_corr[i],
            text=a,
            align='center',
            showarrow=False,
            yanchor='bottom',
            textangle=90) for i,a in enumerate(modelnames)],
            xaxis_title = 'male',
            yaxis_title = 'female',
            )

fig.update_layout(
    shapes=[
        dict(
            type="line",
            x0=0, y0=0,
            x1=1, y1=1,
            line=dict(
                color="RoyalBlue",
                width=2
            )
        )
    ]
)

fig.update_layout(title='Male(X) vs Female(Y)')

fig.show()