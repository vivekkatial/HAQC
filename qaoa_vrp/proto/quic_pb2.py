# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: quic.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import circuit_pb2 as circuit__pb2
import circuit_computation_request_pb2 as circuit__computation__request__pb2
import circuit_computation_result_pb2 as circuit__computation__result__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
    name="quic.proto",
    package="",
    syntax="proto3",
    serialized_options=b"\252\002\010Protobuf",
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\nquic.proto\x1a\rcircuit.proto\x1a!circuit_computation_request.proto\x1a circuit_computation_result.proto"\x8b\x01\n\nQUICircuit\x12\x19\n\x07\x63ircuit\x18\x01 \x01(\x0b\x32\x08.Circuit\x12\x31\n\rsaved_request\x18\x02 \x01(\x0b\x32\x1a.CircuitComputationRequest\x12/\n\x0csaved_result\x18\x03 \x01(\x0b\x32\x19.CircuitComputationResultB\x0b\xaa\x02\x08Protobufb\x06proto3',
    dependencies=[
        circuit__pb2.DESCRIPTOR,
        circuit__computation__request__pb2.DESCRIPTOR,
        circuit__computation__result__pb2.DESCRIPTOR,
    ],
)


_QUICIRCUIT = _descriptor.Descriptor(
    name="QUICircuit",
    full_name="QUICircuit",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="circuit",
            full_name="QUICircuit.circuit",
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
            name="saved_request",
            full_name="QUICircuit.saved_request",
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
            name="saved_result",
            full_name="QUICircuit.saved_result",
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
    serialized_start=99,
    serialized_end=238,
)

_QUICIRCUIT.fields_by_name["circuit"].message_type = circuit__pb2._CIRCUIT
_QUICIRCUIT.fields_by_name[
    "saved_request"
].message_type = circuit__computation__request__pb2._CIRCUITCOMPUTATIONREQUEST
_QUICIRCUIT.fields_by_name[
    "saved_result"
].message_type = circuit__computation__result__pb2._CIRCUITCOMPUTATIONRESULT
DESCRIPTOR.message_types_by_name["QUICircuit"] = _QUICIRCUIT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

QUICircuit = _reflection.GeneratedProtocolMessageType(
    "QUICircuit",
    (_message.Message,),
    {
        "DESCRIPTOR": _QUICIRCUIT,
        "__module__": "quic_pb2"
        # @@protoc_insertion_point(class_scope:QUICircuit)
    },
)
_sym_db.RegisterMessage(QUICircuit)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
