# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: circuit.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import proto.error_pb2 as error__pb2
import proto.param_string_pb2 as param__string__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
    name="circuit.proto",
    package="",
    syntax="proto3",
    serialized_options=b"\252\002\010Protobuf",
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\rcircuit.proto\x1a\x0b\x65rror.proto\x1a\x12param_string.proto"\xc6\x02\n\x07\x43ircuit\x12\n\n\x02id\x18\x01 \x01(\t\x12\x12\n\nnum_qubits\x18\x02 \x01(\x05\x12\x1f\n\x0btime_blocks\x18\x03 \x03(\x0b\x32\n.TimeBlock\x12\'\n\x12global_error_model\x18\x04 \x01(\x0b\x32\x0b.ErrorModel\x12:\n\x12qubit_error_models\x18\x05 \x03(\x0b\x32\x1e.Circuit.QubitErrorModelsEntry\x12#\n\rglobal_params\x18\x06 \x03(\x0b\x32\x0c.GlobalParam\x12*\n\x11time_block_groups\x18\x07 \x03(\x0b\x32\x0f.TimeBlockGroup\x1a\x44\n\x15QubitErrorModelsEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\x1a\n\x05value\x18\x02 \x01(\x0b\x32\x0b.ErrorModel:\x02\x38\x01"7\n\tTimeBlock\x12\n\n\x02id\x18\x01 \x01(\t\x12\x1e\n\ncomponents\x18\x02 \x03(\x0b\x32\n.Component"j\n\x0eTimeBlockGroup\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x1c\n\x14start_time_block_idx\x18\x02 \x01(\x05\x12\x1a\n\x12\x65nd_time_block_idx\x18\x03 \x01(\x05\x12\x10\n\x08\x65xpanded\x18\x04 \x01(\x08"\xfa\x01\n\tComponent\x12\n\n\x02id\x18\x01 \x01(\t\x12&\n\x04type\x18\x02 \x01(\x0e\x32\x18.Component.ComponentType\x12\x13\n\x04gate\x18\x03 \x01(\x0b\x32\x05.Gate\x12\x19\n\x07\x63ircuit\x18\x04 \x01(\x0b\x32\x08.Circuit\x12\x13\n\x0b\x63ircuit_ref\x18\x05 \x01(\t\x12\x0e\n\x06qubits\x18\x06 \x03(\x05\x12\x15\n\rcontrols_zero\x18\x07 \x03(\x05\x12\x14\n\x0c\x63ontrols_one\x18\x08 \x03(\x05"7\n\rComponentType\x12\x08\n\x04GATE\x10\x00\x12\x0b\n\x07\x43IRCUIT\x10\x01\x12\x0f\n\x0b\x43IRCUIT_REF\x10\x02"\xd8\x02\n\x04Gate\x12\x1c\n\x04type\x18\x01 \x01(\x0e\x32\x0e.Gate.GateType\x12\x19\n\x11\x61rb_legacy_params\x18\x02 \x03(\x02\x12#\n\narb_params\x18\x03 \x01(\x0b\x32\x0f.Gate.ArbParams\x1as\n\tArbParams\x12\x1e\n\x08xyz_axes\x18\x01 \x01(\x0b\x32\x0c.InputValue3\x12#\n\x0erotation_angle\x18\x02 \x01(\x0b\x32\x0b.InputValue\x12!\n\x0cglobal_phase\x18\x03 \x01(\x0b\x32\x0b.InputValue"}\n\x08GateType\x12\x0b\n\x07MEASURE\x10\x00\x12\x05\n\x01H\x10\x01\x12\x05\n\x01X\x10\x02\x12\x06\n\x02SX\x10\x03\x12\x05\n\x01Y\x10\x04\x12\x06\n\x02SY\x10\x05\x12\x05\n\x01Z\x10\x06\x12\x05\n\x01S\x10\x07\x12\x05\n\x01T\x10\x08\x12\x07\n\x03NOT\x10\t\x12\x0e\n\nARB_LEGACY\x10\n\x12\x08\n\x04SWAP\x10\x0b\x12\x07\n\x03\x41RB\x10\x0c\x42\x0b\xaa\x02\x08Protobufb\x06proto3',
    dependencies=[
        error__pb2.DESCRIPTOR,
        param__string__pb2.DESCRIPTOR,
    ],
)


_COMPONENT_COMPONENTTYPE = _descriptor.EnumDescriptor(
    name="ComponentType",
    full_name="Component.ComponentType",
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
        _descriptor.EnumValueDescriptor(
            name="GATE",
            index=0,
            number=0,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="CIRCUIT",
            index=1,
            number=1,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="CIRCUIT_REF",
            index=2,
            number=2,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=740,
    serialized_end=795,
)
_sym_db.RegisterEnumDescriptor(_COMPONENT_COMPONENTTYPE)

_GATE_GATETYPE = _descriptor.EnumDescriptor(
    name="GateType",
    full_name="Gate.GateType",
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
        _descriptor.EnumValueDescriptor(
            name="MEASURE",
            index=0,
            number=0,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="H",
            index=1,
            number=1,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="X",
            index=2,
            number=2,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="SX",
            index=3,
            number=3,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="Y",
            index=4,
            number=4,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="SY",
            index=5,
            number=5,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="Z",
            index=6,
            number=6,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="S",
            index=7,
            number=7,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="T",
            index=8,
            number=8,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="NOT",
            index=9,
            number=9,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="ARB_LEGACY",
            index=10,
            number=10,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="SWAP",
            index=11,
            number=11,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="ARB",
            index=12,
            number=12,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=1017,
    serialized_end=1142,
)
_sym_db.RegisterEnumDescriptor(_GATE_GATETYPE)


_CIRCUIT_QUBITERRORMODELSENTRY = _descriptor.Descriptor(
    name="QubitErrorModelsEntry",
    full_name="Circuit.QubitErrorModelsEntry",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="key",
            full_name="Circuit.QubitErrorModelsEntry.key",
            index=0,
            number=1,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="value",
            full_name="Circuit.QubitErrorModelsEntry.value",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=b"8\001",
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=309,
    serialized_end=377,
)

_CIRCUIT = _descriptor.Descriptor(
    name="Circuit",
    full_name="Circuit",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="id",
            full_name="Circuit.id",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="num_qubits",
            full_name="Circuit.num_qubits",
            index=1,
            number=2,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="time_blocks",
            full_name="Circuit.time_blocks",
            index=2,
            number=3,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="global_error_model",
            full_name="Circuit.global_error_model",
            index=3,
            number=4,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="qubit_error_models",
            full_name="Circuit.qubit_error_models",
            index=4,
            number=5,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="global_params",
            full_name="Circuit.global_params",
            index=5,
            number=6,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="time_block_groups",
            full_name="Circuit.time_block_groups",
            index=6,
            number=7,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[
        _CIRCUIT_QUBITERRORMODELSENTRY,
    ],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=51,
    serialized_end=377,
)


_TIMEBLOCK = _descriptor.Descriptor(
    name="TimeBlock",
    full_name="TimeBlock",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="id",
            full_name="TimeBlock.id",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="components",
            full_name="TimeBlock.components",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=379,
    serialized_end=434,
)


_TIMEBLOCKGROUP = _descriptor.Descriptor(
    name="TimeBlockGroup",
    full_name="TimeBlockGroup",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="name",
            full_name="TimeBlockGroup.name",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="start_time_block_idx",
            full_name="TimeBlockGroup.start_time_block_idx",
            index=1,
            number=2,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="end_time_block_idx",
            full_name="TimeBlockGroup.end_time_block_idx",
            index=2,
            number=3,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="expanded",
            full_name="TimeBlockGroup.expanded",
            index=3,
            number=4,
            type=8,
            cpp_type=7,
            label=1,
            has_default_value=False,
            default_value=False,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=436,
    serialized_end=542,
)


_COMPONENT = _descriptor.Descriptor(
    name="Component",
    full_name="Component",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="id",
            full_name="Component.id",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="type",
            full_name="Component.type",
            index=1,
            number=2,
            type=14,
            cpp_type=8,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="gate",
            full_name="Component.gate",
            index=2,
            number=3,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="circuit",
            full_name="Component.circuit",
            index=3,
            number=4,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="circuit_ref",
            full_name="Component.circuit_ref",
            index=4,
            number=5,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="qubits",
            full_name="Component.qubits",
            index=5,
            number=6,
            type=5,
            cpp_type=1,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="controls_zero",
            full_name="Component.controls_zero",
            index=6,
            number=7,
            type=5,
            cpp_type=1,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="controls_one",
            full_name="Component.controls_one",
            index=7,
            number=8,
            type=5,
            cpp_type=1,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[
        _COMPONENT_COMPONENTTYPE,
    ],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=545,
    serialized_end=795,
)


_GATE_ARBPARAMS = _descriptor.Descriptor(
    name="ArbParams",
    full_name="Gate.ArbParams",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="xyz_axes",
            full_name="Gate.ArbParams.xyz_axes",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="rotation_angle",
            full_name="Gate.ArbParams.rotation_angle",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="global_phase",
            full_name="Gate.ArbParams.global_phase",
            index=2,
            number=3,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=900,
    serialized_end=1015,
)

_GATE = _descriptor.Descriptor(
    name="Gate",
    full_name="Gate",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="type",
            full_name="Gate.type",
            index=0,
            number=1,
            type=14,
            cpp_type=8,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="arb_legacy_params",
            full_name="Gate.arb_legacy_params",
            index=1,
            number=2,
            type=2,
            cpp_type=6,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="arb_params",
            full_name="Gate.arb_params",
            index=2,
            number=3,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[
        _GATE_ARBPARAMS,
    ],
    enum_types=[
        _GATE_GATETYPE,
    ],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=798,
    serialized_end=1142,
)

_CIRCUIT_QUBITERRORMODELSENTRY.fields_by_name[
    "value"
].message_type = error__pb2._ERRORMODEL
_CIRCUIT_QUBITERRORMODELSENTRY.containing_type = _CIRCUIT
_CIRCUIT.fields_by_name["time_blocks"].message_type = _TIMEBLOCK
_CIRCUIT.fields_by_name["global_error_model"].message_type = error__pb2._ERRORMODEL
_CIRCUIT.fields_by_name[
    "qubit_error_models"
].message_type = _CIRCUIT_QUBITERRORMODELSENTRY
_CIRCUIT.fields_by_name["global_params"].message_type = param__string__pb2._GLOBALPARAM
_CIRCUIT.fields_by_name["time_block_groups"].message_type = _TIMEBLOCKGROUP
_TIMEBLOCK.fields_by_name["components"].message_type = _COMPONENT
_COMPONENT.fields_by_name["type"].enum_type = _COMPONENT_COMPONENTTYPE
_COMPONENT.fields_by_name["gate"].message_type = _GATE
_COMPONENT.fields_by_name["circuit"].message_type = _CIRCUIT
_COMPONENT_COMPONENTTYPE.containing_type = _COMPONENT
_GATE_ARBPARAMS.fields_by_name[
    "xyz_axes"
].message_type = param__string__pb2._INPUTVALUE3
_GATE_ARBPARAMS.fields_by_name[
    "rotation_angle"
].message_type = param__string__pb2._INPUTVALUE
_GATE_ARBPARAMS.fields_by_name[
    "global_phase"
].message_type = param__string__pb2._INPUTVALUE
_GATE_ARBPARAMS.containing_type = _GATE
_GATE.fields_by_name["type"].enum_type = _GATE_GATETYPE
_GATE.fields_by_name["arb_params"].message_type = _GATE_ARBPARAMS
_GATE_GATETYPE.containing_type = _GATE
DESCRIPTOR.message_types_by_name["Circuit"] = _CIRCUIT
DESCRIPTOR.message_types_by_name["TimeBlock"] = _TIMEBLOCK
DESCRIPTOR.message_types_by_name["TimeBlockGroup"] = _TIMEBLOCKGROUP
DESCRIPTOR.message_types_by_name["Component"] = _COMPONENT
DESCRIPTOR.message_types_by_name["Gate"] = _GATE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Circuit = _reflection.GeneratedProtocolMessageType(
    "Circuit",
    (_message.Message,),
    {
        "QubitErrorModelsEntry": _reflection.GeneratedProtocolMessageType(
            "QubitErrorModelsEntry",
            (_message.Message,),
            {
                "DESCRIPTOR": _CIRCUIT_QUBITERRORMODELSENTRY,
                "__module__": "circuit_pb2"
                # @@protoc_insertion_point(class_scope:Circuit.QubitErrorModelsEntry)
            },
        ),
        "DESCRIPTOR": _CIRCUIT,
        "__module__": "circuit_pb2"
        # @@protoc_insertion_point(class_scope:Circuit)
    },
)
_sym_db.RegisterMessage(Circuit)
_sym_db.RegisterMessage(Circuit.QubitErrorModelsEntry)

TimeBlock = _reflection.GeneratedProtocolMessageType(
    "TimeBlock",
    (_message.Message,),
    {
        "DESCRIPTOR": _TIMEBLOCK,
        "__module__": "circuit_pb2"
        # @@protoc_insertion_point(class_scope:TimeBlock)
    },
)
_sym_db.RegisterMessage(TimeBlock)

TimeBlockGroup = _reflection.GeneratedProtocolMessageType(
    "TimeBlockGroup",
    (_message.Message,),
    {
        "DESCRIPTOR": _TIMEBLOCKGROUP,
        "__module__": "circuit_pb2"
        # @@protoc_insertion_point(class_scope:TimeBlockGroup)
    },
)
_sym_db.RegisterMessage(TimeBlockGroup)

Component = _reflection.GeneratedProtocolMessageType(
    "Component",
    (_message.Message,),
    {
        "DESCRIPTOR": _COMPONENT,
        "__module__": "circuit_pb2"
        # @@protoc_insertion_point(class_scope:Component)
    },
)
_sym_db.RegisterMessage(Component)

Gate = _reflection.GeneratedProtocolMessageType(
    "Gate",
    (_message.Message,),
    {
        "ArbParams": _reflection.GeneratedProtocolMessageType(
            "ArbParams",
            (_message.Message,),
            {
                "DESCRIPTOR": _GATE_ARBPARAMS,
                "__module__": "circuit_pb2"
                # @@protoc_insertion_point(class_scope:Gate.ArbParams)
            },
        ),
        "DESCRIPTOR": _GATE,
        "__module__": "circuit_pb2"
        # @@protoc_insertion_point(class_scope:Gate)
    },
)
_sym_db.RegisterMessage(Gate)
_sym_db.RegisterMessage(Gate.ArbParams)


DESCRIPTOR._options = None
_CIRCUIT_QUBITERRORMODELSENTRY._options = None
# @@protoc_insertion_point(module_scope)
