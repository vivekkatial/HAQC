# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: float3.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name="float3.proto",
    package="",
    syntax="proto3",
    serialized_options=b"\252\002\010Protobuf",
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x0c\x66loat3.proto")\n\x06\x46loat3\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\x42\x0b\xaa\x02\x08Protobufb\x06proto3',
)


_FLOAT3 = _descriptor.Descriptor(
    name="Float3",
    full_name="Float3",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="x",
            full_name="Float3.x",
            index=0,
            number=1,
            type=2,
            cpp_type=6,
            label=1,
            has_default_value=False,
            default_value=float(0),
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
            name="y",
            full_name="Float3.y",
            index=1,
            number=2,
            type=2,
            cpp_type=6,
            label=1,
            has_default_value=False,
            default_value=float(0),
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
            name="z",
            full_name="Float3.z",
            index=2,
            number=3,
            type=2,
            cpp_type=6,
            label=1,
            has_default_value=False,
            default_value=float(0),
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
    serialized_start=16,
    serialized_end=57,
)

DESCRIPTOR.message_types_by_name["Float3"] = _FLOAT3
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Float3 = _reflection.GeneratedProtocolMessageType(
    "Float3",
    (_message.Message,),
    {
        "DESCRIPTOR": _FLOAT3,
        "__module__": "float3_pb2"
        # @@protoc_insertion_point(class_scope:Float3)
    },
)
_sym_db.RegisterMessage(Float3)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)