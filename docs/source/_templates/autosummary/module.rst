{{ name | escape | underline}}

.. automodule:: {{ fullname }}

{% block attributes %}
    {% if attributes %}
        .. rubric:: {{ _('Module Attributes') }}
        .. autosummary::
            {% for item in attributes %}
                {{ item }}
            {%- endfor %}
        {% endif %}
{% endblock %}

{% block functions %}
    {% if functions %}
        .. rubric:: {{ _('Functions') }}
        .. autosummary::
            :toctree:
            :recursive:
            {% if fullname == 'simphony.defaults' %}
            {% set preferred_functions = ['default_nv_model', 'default_multi_nv_model', 'default_nv_hyperfine_database'] %}
            {% for item in preferred_functions %}
                {% if item in functions %}
                {{ item }}
                {% endif %}
            {% endfor %}
            {% for item in functions %}
                {% if item not in preferred_functions %}
                {{ item }}
                {% endif %}
            {% endfor %}
            {% else %}
            {% for item in functions %}
                {{ item }}
            {%- endfor %}
            {% endif %}
    {% endif %}
{% endblock %}

{% block classes %}
    {% if classes %}
        {% if fullname == 'simphony.components' %}
        {% set core_classes = ['ElectronSpin', 'NuclearSpin', 'Interaction', 'StaticField', 'LinearDrivingField', 'CircularDrivingField'] %}
        {% set low_api_classes = ['BaseSpin', 'Pulse', 'PulseList', 'Signal', 'SignalSum', 'TimeSegment', 'TimeSegmentSequence', 'PlotSpec', 'SimulationSpecBase', 'SimulationSpecBasic', 'SimulationSpecSingleSineWave'] %}
        {% set ordered_all = core_classes + low_api_classes %}
        .. rubric:: {{ _('Classes') }}
        .. autosummary::
            :toctree:
            :recursive:

            {% for item in core_classes %}
                {% if item in classes %}
                {{ item }}
                {% endif %}
            {% endfor %}

        .. rubric:: Low API Classes
        .. autosummary::
            :toctree:
            :recursive:

            {% for item in low_api_classes %}
                {% if item in classes %}
                {{ item }}
                {% endif %}
            {% endfor %}
            {% for item in classes %}
                {% if item not in ordered_all %}
                {{ item }}
                {% endif %}
            {% endfor %}
        {% elif fullname == 'simphony.model' %}
        {% set preferred_classes = ['Model', 'RotatingFrameSetter'] %}
        .. rubric:: {{ _('Classes') }}
        .. autosummary::
            :toctree:
            :recursive:
            {% for item in preferred_classes %}
                {% if item in classes %}
                {{ item }}
                {% endif %}
            {% endfor %}
            {% for item in classes %}
                {% if item not in preferred_classes %}
                {{ item }}
                {% endif %}
            {% endfor %}
        {% else %}
        .. rubric:: {{ _('Classes') }}
        .. autosummary::
            :toctree:
            :recursive:
            {% for item in classes %}
                {{ item }}
            {%- endfor %}
        {% endif %}
    {% endif %}
{% endblock %}

{% block exceptions %}
    {% if exceptions %}
        .. rubric:: {{ _('Exceptions') }}
        .. autosummary::
            :toctree:
            :recursive:
            {% for item in exceptions %}
                {{ item }}
            {%- endfor %}
    {% endif %}
{% endblock %}

{% block modules %}
    {% if modules %}
        .. rubric:: Modules
        .. autosummary::
            :toctree:
            :recursive:
            {% for item in modules %}
                {{ item }}
            {%- endfor %}
    {% endif %}
{% endblock %}
