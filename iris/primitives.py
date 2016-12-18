from .iris_types_new import IrisValue, IrisImage, Int, IrisType, Any, List, String

# write matplotlib plot to bytes encoded as string
def send_plot(plt, name):
    import io
    import base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return IrisImage(base64.b64encode(buf.read()).decode('utf-8'), name.id, name.name)
