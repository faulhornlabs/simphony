{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}



{% block methods %}

{%- set public_methods = [] %}

{% for item in methods %}
    {% if not item.startswith('_') or item == '__call__' %}
        {%- set _ = public_methods.append(item) %}
    {% endif %}
{% endfor %}

{% if public_methods %}
    .. rubric:: {{ _('Methods') }}
    .. autosummary::
        :toctree:
        :recursive:
        {% for item in public_methods %}
            ~{{ name }}.{{ item }}
        {%- endfor %}
{% endif %}

{% endblock %}



{% block attributes %}

{%- set shown_attributes = [] %}

{% for item in attributes | sort %}
    {%- set _ = shown_attributes.append(item) %}
{%- endfor %}

{% if shown_attributes %}
    .. rubric:: {{ _('Attributes') }}
    .. autosummary::
        :toctree:
        :recursive:
        {% for item in shown_attributes %}
            ~{{ name }}.{{ item }}
        {%- endfor %}
    {% endif %}

{% endblock %}
