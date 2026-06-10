.. _{{ fullname | replace('.', '_') }}:

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

{% if module == 'simphony.model' and objname == 'Model' %}
    {% set model_items = [
        ('Overview', [
            'name',
            'n_spins',
            'dimension',
            'subdimensions',
            'spin_names',
            'driving_field_names',
        ]),
        ('Components', [
            'spins',
            'spin',
            'add_spin',
            'remove_spin',
            'interactions',
            'add_interaction',
            'remove_interaction',
            'static_fields',
            'add_static_field',
            'remove_static_field',
            'driving_fields',
            'driving_field',
            'add_driving_field',
            'remove_driving_field',
        ]),
        ('Bases and Eigensystem', [
            'basis',
            'basis_qubit_subspace',
            'state',
            'productstate',
            'eigenstate',
            'eigenenergy',
            'eigenbasis',
            'eigenenergies',
        ]),
        ('Hamiltonians, Operators and Frames', [
            'static_hamiltonian',
            'driving_operators',
            'local_quasistatic_noise_operators',
            'rotating_frame',
            'rotating_frame_frequencies',
            'rotating_frame_hamiltonian',
            'rotating_frame_operator',
            'virtual_phases',
            'virtual_phases_operator',
            'operator_from_string',
        ]),
        ('Transitions and Control', [
            'splitting',
            'splitting_qubit',
            'rabi_amplitude',
            'rabi_amplitude_qubit',
            'rabi_period',
            'rabi_period_qubit',
            'matrix_element',
        ]),
        ('Simulation', [
            'simulate_time_evolution',
            'initial_state',
            'pulses',
            'last_pulse_end',
            'remove_all_pulses',
        ]),
        ('Visualization', [
            'plot_driving_fields',
            'plot_levels',
        ]),
        ('Utils', [
            'project_to_qubit_subspace',
            'test_labeling',
        ]),
    ] %}

    {%- set sidebar_items = [] %}
    {% for title, items in model_items %}
        {% for item in items %}
            {% if (item in methods or item in attributes) and item not in sidebar_items %}
                {%- set _ = sidebar_items.append(item) %}
            {% endif %}
        {% endfor %}
    {% endfor %}

    .. container:: sidebar-only

        .. autosummary::
            :toctree:
            :recursive:
            :nosignatures:
            {% for item in sidebar_items | sort %}
                ~{{ name }}.{{ item }}
            {%- endfor %}

    {% for title, items in model_items %}
    {%- set grouped_items = [] %}
    {% for item in items %}
        {% if item in methods or item in attributes %}
            {%- set _ = grouped_items.append(item) %}
        {% endif %}
    {% endfor %}
    {% if grouped_items %}
    .. rubric:: {{ title }}
    .. autosummary::
        {% for item in grouped_items %}
            ~{{ name }}.{{ item }}
        {%- endfor %}
    {% endif %}
    {% endfor %}
{% elif public_methods and not (module == 'simphony.defaults' and objname in ['NVLatticeSite', 'NVLatticeDatabase']) %}
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

{% if module == 'simphony.model' and objname == 'Model' %}
{% elif shown_attributes and not (module == 'simphony.defaults' and objname in ['NVLatticeSite', 'NVLatticeDatabase']) %}
    .. rubric:: {{ _('Attributes') }}
    .. autosummary::
        :toctree:
        :recursive:
        {% for item in shown_attributes %}
            ~{{ name }}.{{ item }}
        {%- endfor %}
{% endif %}

{% endblock %}
