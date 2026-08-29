import datetime

from google.protobuf import duration_pb2 as _duration_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class JobState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    JOB_STATE_UNSPECIFIED: _ClassVar[JobState]
    JOB_STATE_PENDING: _ClassVar[JobState]
    JOB_STATE_RUNNING: _ClassVar[JobState]
    JOB_STATE_SUCCEEDED: _ClassVar[JobState]
    JOB_STATE_FAILED: _ClassVar[JobState]
    JOB_STATE_CANCELLED: _ClassVar[JobState]

class ChainState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CHAIN_STATE_UNSPECIFIED: _ClassVar[ChainState]
    CHAIN_STATE_ACTIVE: _ClassVar[ChainState]
    CHAIN_STATE_DRAINING: _ClassVar[ChainState]
JOB_STATE_UNSPECIFIED: JobState
JOB_STATE_PENDING: JobState
JOB_STATE_RUNNING: JobState
JOB_STATE_SUCCEEDED: JobState
JOB_STATE_FAILED: JobState
JOB_STATE_CANCELLED: JobState
CHAIN_STATE_UNSPECIFIED: ChainState
CHAIN_STATE_ACTIVE: ChainState
CHAIN_STATE_DRAINING: ChainState

class Job(_message.Message):
    __slots__ = ("job_id", "state", "total_chunk_count", "succeeded_chunk_count", "failed_chunk_count", "max_retries", "retry_backoff", "lease_duration", "created_at", "registration_completed_at", "updated_at", "terminal_at")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CHUNK_COUNT_FIELD_NUMBER: _ClassVar[int]
    SUCCEEDED_CHUNK_COUNT_FIELD_NUMBER: _ClassVar[int]
    FAILED_CHUNK_COUNT_FIELD_NUMBER: _ClassVar[int]
    MAX_RETRIES_FIELD_NUMBER: _ClassVar[int]
    RETRY_BACKOFF_FIELD_NUMBER: _ClassVar[int]
    LEASE_DURATION_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    REGISTRATION_COMPLETED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_AT_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    state: JobState
    total_chunk_count: int
    succeeded_chunk_count: int
    failed_chunk_count: int
    max_retries: int
    retry_backoff: _duration_pb2.Duration
    lease_duration: _duration_pb2.Duration
    created_at: _timestamp_pb2.Timestamp
    registration_completed_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    terminal_at: _timestamp_pb2.Timestamp
    def __init__(self, job_id: _Optional[str] = ..., state: _Optional[_Union[JobState, str]] = ..., total_chunk_count: _Optional[int] = ..., succeeded_chunk_count: _Optional[int] = ..., failed_chunk_count: _Optional[int] = ..., max_retries: _Optional[int] = ..., retry_backoff: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., lease_duration: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., registration_completed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., terminal_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ChainIdentity(_message.Message):
    __slots__ = ("job_id", "rank_id", "chain_id")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    RANK_ID_FIELD_NUMBER: _ClassVar[int]
    CHAIN_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    rank_id: str
    chain_id: int
    def __init__(self, job_id: _Optional[str] = ..., rank_id: _Optional[str] = ..., chain_id: _Optional[int] = ...) -> None: ...

class ChainAssociation(_message.Message):
    __slots__ = ("identity", "state", "created_at", "draining_at")
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    DRAINING_AT_FIELD_NUMBER: _ClassVar[int]
    identity: ChainIdentity
    state: ChainState
    created_at: _timestamp_pb2.Timestamp
    draining_at: _timestamp_pb2.Timestamp
    def __init__(self, identity: _Optional[_Union[ChainIdentity, _Mapping]] = ..., state: _Optional[_Union[ChainState, str]] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., draining_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ChunkRegistration(_message.Message):
    __slots__ = ("chunk_id", "input_ref")
    CHUNK_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_REF_FIELD_NUMBER: _ClassVar[int]
    chunk_id: int
    input_ref: str
    def __init__(self, chunk_id: _Optional[int] = ..., input_ref: _Optional[str] = ...) -> None: ...

class ChunkLease(_message.Message):
    __slots__ = ("chunk_id", "input_ref", "generation", "expires_at", "retry_count")
    CHUNK_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_REF_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    RETRY_COUNT_FIELD_NUMBER: _ClassVar[int]
    chunk_id: int
    input_ref: str
    generation: int
    expires_at: _timestamp_pb2.Timestamp
    retry_count: int
    def __init__(self, chunk_id: _Optional[int] = ..., input_ref: _Optional[str] = ..., generation: _Optional[int] = ..., expires_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., retry_count: _Optional[int] = ...) -> None: ...

class LeaseReference(_message.Message):
    __slots__ = ("chunk_id", "generation")
    CHUNK_ID_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    chunk_id: int
    generation: int
    def __init__(self, chunk_id: _Optional[int] = ..., generation: _Optional[int] = ...) -> None: ...

class RenewedLease(_message.Message):
    __slots__ = ("lease", "expires_at")
    LEASE_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    lease: LeaseReference
    expires_at: _timestamp_pb2.Timestamp
    def __init__(self, lease: _Optional[_Union[LeaseReference, _Mapping]] = ..., expires_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class CreateJobRequest(_message.Message):
    __slots__ = ("job_id", "total_chunk_count", "max_retries", "retry_backoff", "lease_duration")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CHUNK_COUNT_FIELD_NUMBER: _ClassVar[int]
    MAX_RETRIES_FIELD_NUMBER: _ClassVar[int]
    RETRY_BACKOFF_FIELD_NUMBER: _ClassVar[int]
    LEASE_DURATION_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    total_chunk_count: int
    max_retries: int
    retry_backoff: _duration_pb2.Duration
    lease_duration: _duration_pb2.Duration
    def __init__(self, job_id: _Optional[str] = ..., total_chunk_count: _Optional[int] = ..., max_retries: _Optional[int] = ..., retry_backoff: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., lease_duration: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ...) -> None: ...

class CreateJobResponse(_message.Message):
    __slots__ = ("job",)
    JOB_FIELD_NUMBER: _ClassVar[int]
    job: Job
    def __init__(self, job: _Optional[_Union[Job, _Mapping]] = ...) -> None: ...

class GetJobRequest(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class GetJobResponse(_message.Message):
    __slots__ = ("job",)
    JOB_FIELD_NUMBER: _ClassVar[int]
    job: Job
    def __init__(self, job: _Optional[_Union[Job, _Mapping]] = ...) -> None: ...

class RegisterChunksRequest(_message.Message):
    __slots__ = ("job_id", "chunks")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    chunks: _containers.RepeatedCompositeFieldContainer[ChunkRegistration]
    def __init__(self, job_id: _Optional[str] = ..., chunks: _Optional[_Iterable[_Union[ChunkRegistration, _Mapping]]] = ...) -> None: ...

class RegisterChunksResponse(_message.Message):
    __slots__ = ("registered_count",)
    REGISTERED_COUNT_FIELD_NUMBER: _ClassVar[int]
    registered_count: int
    def __init__(self, registered_count: _Optional[int] = ...) -> None: ...

class FinalizeJobRegistrationRequest(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class FinalizeJobRegistrationResponse(_message.Message):
    __slots__ = ("job",)
    JOB_FIELD_NUMBER: _ClassVar[int]
    job: Job
    def __init__(self, job: _Optional[_Union[Job, _Mapping]] = ...) -> None: ...

class CancelJobRequest(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class CancelJobResponse(_message.Message):
    __slots__ = ("job",)
    JOB_FIELD_NUMBER: _ClassVar[int]
    job: Job
    def __init__(self, job: _Optional[_Union[Job, _Mapping]] = ...) -> None: ...

class AddChainAssociationRequest(_message.Message):
    __slots__ = ("chain",)
    CHAIN_FIELD_NUMBER: _ClassVar[int]
    chain: ChainIdentity
    def __init__(self, chain: _Optional[_Union[ChainIdentity, _Mapping]] = ...) -> None: ...

class AddChainAssociationResponse(_message.Message):
    __slots__ = ("association",)
    ASSOCIATION_FIELD_NUMBER: _ClassVar[int]
    association: ChainAssociation
    def __init__(self, association: _Optional[_Union[ChainAssociation, _Mapping]] = ...) -> None: ...

class DrainChainAssociationRequest(_message.Message):
    __slots__ = ("chain",)
    CHAIN_FIELD_NUMBER: _ClassVar[int]
    chain: ChainIdentity
    def __init__(self, chain: _Optional[_Union[ChainIdentity, _Mapping]] = ...) -> None: ...

class DrainChainAssociationResponse(_message.Message):
    __slots__ = ("association",)
    ASSOCIATION_FIELD_NUMBER: _ClassVar[int]
    association: ChainAssociation
    def __init__(self, association: _Optional[_Union[ChainAssociation, _Mapping]] = ...) -> None: ...

class ClaimChunksRequest(_message.Message):
    __slots__ = ("chain", "max_chunks")
    CHAIN_FIELD_NUMBER: _ClassVar[int]
    MAX_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    chain: ChainIdentity
    max_chunks: int
    def __init__(self, chain: _Optional[_Union[ChainIdentity, _Mapping]] = ..., max_chunks: _Optional[int] = ...) -> None: ...

class ClaimChunksResponse(_message.Message):
    __slots__ = ("job_state", "leases", "database_time")
    JOB_STATE_FIELD_NUMBER: _ClassVar[int]
    LEASES_FIELD_NUMBER: _ClassVar[int]
    DATABASE_TIME_FIELD_NUMBER: _ClassVar[int]
    job_state: JobState
    leases: _containers.RepeatedCompositeFieldContainer[ChunkLease]
    database_time: _timestamp_pb2.Timestamp
    def __init__(self, job_state: _Optional[_Union[JobState, str]] = ..., leases: _Optional[_Iterable[_Union[ChunkLease, _Mapping]]] = ..., database_time: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class RenewLeasesRequest(_message.Message):
    __slots__ = ("chain", "leases")
    CHAIN_FIELD_NUMBER: _ClassVar[int]
    LEASES_FIELD_NUMBER: _ClassVar[int]
    chain: ChainIdentity
    leases: _containers.RepeatedCompositeFieldContainer[LeaseReference]
    def __init__(self, chain: _Optional[_Union[ChainIdentity, _Mapping]] = ..., leases: _Optional[_Iterable[_Union[LeaseReference, _Mapping]]] = ...) -> None: ...

class RenewLeasesResponse(_message.Message):
    __slots__ = ("renewed", "stale", "database_time")
    RENEWED_FIELD_NUMBER: _ClassVar[int]
    STALE_FIELD_NUMBER: _ClassVar[int]
    DATABASE_TIME_FIELD_NUMBER: _ClassVar[int]
    renewed: _containers.RepeatedCompositeFieldContainer[RenewedLease]
    stale: _containers.RepeatedCompositeFieldContainer[LeaseReference]
    database_time: _timestamp_pb2.Timestamp
    def __init__(self, renewed: _Optional[_Iterable[_Union[RenewedLease, _Mapping]]] = ..., stale: _Optional[_Iterable[_Union[LeaseReference, _Mapping]]] = ..., database_time: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class CompleteChunkRequest(_message.Message):
    __slots__ = ("chain", "lease", "output_uri", "checksum", "output_size_bytes")
    CHAIN_FIELD_NUMBER: _ClassVar[int]
    LEASE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_URI_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    chain: ChainIdentity
    lease: LeaseReference
    output_uri: str
    checksum: str
    output_size_bytes: int
    def __init__(self, chain: _Optional[_Union[ChainIdentity, _Mapping]] = ..., lease: _Optional[_Union[LeaseReference, _Mapping]] = ..., output_uri: _Optional[str] = ..., checksum: _Optional[str] = ..., output_size_bytes: _Optional[int] = ...) -> None: ...

class CompleteChunkResponse(_message.Message):
    __slots__ = ("job_state", "replayed")
    JOB_STATE_FIELD_NUMBER: _ClassVar[int]
    REPLAYED_FIELD_NUMBER: _ClassVar[int]
    job_state: JobState
    replayed: bool
    def __init__(self, job_state: _Optional[_Union[JobState, str]] = ..., replayed: _Optional[bool] = ...) -> None: ...

class FailChunkRequest(_message.Message):
    __slots__ = ("chain", "lease", "failure_class", "message", "retriable")
    CHAIN_FIELD_NUMBER: _ClassVar[int]
    LEASE_FIELD_NUMBER: _ClassVar[int]
    FAILURE_CLASS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RETRIABLE_FIELD_NUMBER: _ClassVar[int]
    chain: ChainIdentity
    lease: LeaseReference
    failure_class: str
    message: str
    retriable: bool
    def __init__(self, chain: _Optional[_Union[ChainIdentity, _Mapping]] = ..., lease: _Optional[_Union[LeaseReference, _Mapping]] = ..., failure_class: _Optional[str] = ..., message: _Optional[str] = ..., retriable: _Optional[bool] = ...) -> None: ...

class FailChunkResponse(_message.Message):
    __slots__ = ("job_state",)
    JOB_STATE_FIELD_NUMBER: _ClassVar[int]
    job_state: JobState
    def __init__(self, job_state: _Optional[_Union[JobState, str]] = ...) -> None: ...
