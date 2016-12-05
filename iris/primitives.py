from .iris_types import IrisValue, IrisImage, Int, IrisType, Any, List, String

# write matplotlib plot to bytes encoded as string
def send_plot(plt):
    import io
    import base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return IrisImage(value=base64.b64encode(buf.read()).decode('utf-8'))
