class ResponsibilityBars:
    # Takes a weighting, plots all slices on x-axis and a bar with the height being the weighting factor for that particular slice
    # optionally adds an h-line for a uniform weighting reference line

    # old code
    # def get_fig(self, result: SelectionResult, show_equal: bool = True, title: str | None = None) -> go.Figure:
    #     data = pd.DataFrame({"slice": [str(s) for s in result.responsibility_weights.keys()], "responsibility_share": list(result.responsibility_weights.values())})
    #     fig = px.bar(data, x="slice", y="responsibility_share", title=title or "Coverage responsibility per selected slice", text="responsibility_share")
    #     fig.update_traces(texttemplate="%{y:.3f}", textposition="outside")
    #     fig.update_yaxes(range=[0, 1], title="share")
    #     if show_equal and len(data) > 0:
    #         fig.add_hline(y=1.0 / len(data), line_dash="dot", annotation_text="equal weight", annotation_position="top left")
    #     return fig

    pass