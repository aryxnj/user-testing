import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

def scroll_to_top():
    components.html(
        """
        <script>
        window.scrollTo({ top: 0, behavior: 'smooth' });
        </script>
        """,
        height=1,
        width=1,
        key=f"scroll_{datetime.now().timestamp()}"
    )

# Call scroll_to_top
scroll_to_top()

st.write("If you see this, the scroll_to_top function executed without errors.")
