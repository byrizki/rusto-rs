package rusto

// #include <rusto.h>
import "C"

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"runtime"
	"sync/atomic"
	"unsafe"
)

// This is needed, because as of go 1.24
// type RustBuffer C.RustBuffer cannot have methods,
// RustBuffer is treated as non-local type
type GoRustBuffer struct {
	inner C.RustBuffer
}

type RustBufferI interface {
	AsReader() *bytes.Reader
	Free()
	ToGoBytes() []byte
	Data() unsafe.Pointer
	Len() uint64
	Capacity() uint64
}

// C.RustBuffer fields exposed as an interface so they can be accessed in different Go packages.
// See https://github.com/golang/go/issues/13467
type ExternalCRustBuffer interface {
	Data() unsafe.Pointer
	Len() uint64
	Capacity() uint64
}

func RustBufferFromC(b C.RustBuffer) ExternalCRustBuffer {
	return GoRustBuffer{
		inner: b,
	}
}

func CFromRustBuffer(b ExternalCRustBuffer) C.RustBuffer {
	return C.RustBuffer{
		capacity: C.uint64_t(b.Capacity()),
		len:      C.uint64_t(b.Len()),
		data:     (*C.uchar)(b.Data()),
	}
}

func RustBufferFromExternal(b ExternalCRustBuffer) GoRustBuffer {
	return GoRustBuffer{
		inner: C.RustBuffer{
			capacity: C.uint64_t(b.Capacity()),
			len:      C.uint64_t(b.Len()),
			data:     (*C.uchar)(b.Data()),
		},
	}
}

func (cb GoRustBuffer) Capacity() uint64 {
	return uint64(cb.inner.capacity)
}

func (cb GoRustBuffer) Len() uint64 {
	return uint64(cb.inner.len)
}

func (cb GoRustBuffer) Data() unsafe.Pointer {
	return unsafe.Pointer(cb.inner.data)
}

func (cb GoRustBuffer) AsReader() *bytes.Reader {
	b := unsafe.Slice((*byte)(cb.inner.data), C.uint64_t(cb.inner.len))
	return bytes.NewReader(b)
}

func (cb GoRustBuffer) Free() {
	rustCall(func(status *C.RustCallStatus) bool {
		C.ffi_rusto_rustbuffer_free(cb.inner, status)
		return false
	})
}

func (cb GoRustBuffer) ToGoBytes() []byte {
	return C.GoBytes(unsafe.Pointer(cb.inner.data), C.int(cb.inner.len))
}

func stringToRustBuffer(str string) C.RustBuffer {
	return bytesToRustBuffer([]byte(str))
}

func bytesToRustBuffer(b []byte) C.RustBuffer {
	if len(b) == 0 {
		return C.RustBuffer{}
	}
	// We can pass the pointer along here, as it is pinned
	// for the duration of this call
	foreign := C.ForeignBytes{
		len:  C.int(len(b)),
		data: (*C.uchar)(unsafe.Pointer(&b[0])),
	}

	return rustCall(func(status *C.RustCallStatus) C.RustBuffer {
		return C.ffi_rusto_rustbuffer_from_bytes(foreign, status)
	})
}

type BufLifter[GoType any] interface {
	Lift(value RustBufferI) GoType
}

type BufLowerer[GoType any] interface {
	Lower(value GoType) C.RustBuffer
}

type BufReader[GoType any] interface {
	Read(reader io.Reader) GoType
}

type BufWriter[GoType any] interface {
	Write(writer io.Writer, value GoType)
}

func LowerIntoRustBuffer[GoType any](bufWriter BufWriter[GoType], value GoType) C.RustBuffer {
	// This might be not the most efficient way but it does not require knowing allocation size
	// beforehand
	var buffer bytes.Buffer
	bufWriter.Write(&buffer, value)

	bytes, err := io.ReadAll(&buffer)
	if err != nil {
		panic(fmt.Errorf("reading written data: %w", err))
	}
	return bytesToRustBuffer(bytes)
}

func LiftFromRustBuffer[GoType any](bufReader BufReader[GoType], rbuf RustBufferI) GoType {
	defer rbuf.Free()
	reader := rbuf.AsReader()
	item := bufReader.Read(reader)
	if reader.Len() > 0 {
		// TODO: Remove this
		leftover, _ := io.ReadAll(reader)
		panic(fmt.Errorf("Junk remaining in buffer after lifting: %s", string(leftover)))
	}
	return item
}

func rustCallWithError[E any, U any](converter BufReader[E], callback func(*C.RustCallStatus) U) (U, E) {
	var status C.RustCallStatus
	returnValue := callback(&status)
	err := checkCallStatus(converter, status)
	return returnValue, err
}

func checkCallStatus[E any](converter BufReader[E], status C.RustCallStatus) E {
	switch status.code {
	case 0:
		var zero E
		return zero
	case 1:
		return LiftFromRustBuffer(converter, GoRustBuffer{inner: status.errorBuf})
	case 2:
		// when the rust code sees a panic, it tries to construct a rustBuffer
		// with the message.  but if that code panics, then it just sends back
		// an empty buffer.
		if status.errorBuf.len > 0 {
			panic(fmt.Errorf("%s", FfiConverterStringINSTANCE.Lift(GoRustBuffer{inner: status.errorBuf})))
		} else {
			panic(fmt.Errorf("Rust panicked while handling Rust panic"))
		}
	default:
		panic(fmt.Errorf("unknown status code: %d", status.code))
	}
}

func checkCallStatusUnknown(status C.RustCallStatus) error {
	switch status.code {
	case 0:
		return nil
	case 1:
		panic(fmt.Errorf("function not returning an error returned an error"))
	case 2:
		// when the rust code sees a panic, it tries to construct a C.RustBuffer
		// with the message.  but if that code panics, then it just sends back
		// an empty buffer.
		if status.errorBuf.len > 0 {
			panic(fmt.Errorf("%s", FfiConverterStringINSTANCE.Lift(GoRustBuffer{
				inner: status.errorBuf,
			})))
		} else {
			panic(fmt.Errorf("Rust panicked while handling Rust panic"))
		}
	default:
		return fmt.Errorf("unknown status code: %d", status.code)
	}
}

func rustCall[U any](callback func(*C.RustCallStatus) U) U {
	returnValue, err := rustCallWithError[error](nil, callback)
	if err != nil {
		panic(err)
	}
	return returnValue
}

type NativeError interface {
	AsError() error
}

func writeInt8(writer io.Writer, value int8) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func writeUint8(writer io.Writer, value uint8) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func writeInt16(writer io.Writer, value int16) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func writeUint16(writer io.Writer, value uint16) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func writeInt32(writer io.Writer, value int32) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func writeUint32(writer io.Writer, value uint32) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func writeInt64(writer io.Writer, value int64) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func writeUint64(writer io.Writer, value uint64) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func writeFloat32(writer io.Writer, value float32) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func writeFloat64(writer io.Writer, value float64) {
	if err := binary.Write(writer, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func readInt8(reader io.Reader) int8 {
	var result int8
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func readUint8(reader io.Reader) uint8 {
	var result uint8
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func readInt16(reader io.Reader) int16 {
	var result int16
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func readUint16(reader io.Reader) uint16 {
	var result uint16
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func readInt32(reader io.Reader) int32 {
	var result int32
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func readUint32(reader io.Reader) uint32 {
	var result uint32
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func readInt64(reader io.Reader) int64 {
	var result int64
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func readUint64(reader io.Reader) uint64 {
	var result uint64
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func readFloat32(reader io.Reader) float32 {
	var result float32
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func readFloat64(reader io.Reader) float64 {
	var result float64
	if err := binary.Read(reader, binary.BigEndian, &result); err != nil {
		panic(err)
	}
	return result
}

func init() {

	uniffiCheckChecksums()
}

func uniffiCheckChecksums() {
	// Get the bindings contract version from our ComponentInterface
	bindingsContractVersion := 30
	// Get the scaffolding contract version by calling the into the dylib
	scaffoldingContractVersion := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint32_t {
		return C.ffi_rusto_uniffi_contract_version()
	})
	if bindingsContractVersion != int(scaffoldingContractVersion) {
		// If this happens try cleaning and rebuilding your project
		panic("rusto: UniFFI contract version mismatch")
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_rusto_checksum_func_create_initialize_config_ppv6()
		})
		if checksum != 12132 {
			// If this happens try cleaning and rebuilding your project
			panic("rusto: uniffi_rusto_checksum_func_create_initialize_config_ppv6: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_rusto_checksum_func_default_ocr_run_options()
		})
		if checksum != 50314 {
			// If this happens try cleaning and rebuilding your project
			panic("rusto: uniffi_rusto_checksum_func_default_ocr_run_options: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_rusto_checksum_func_rusto_version()
		})
		if checksum != 48676 {
			// If this happens try cleaning and rebuilding your project
			panic("rusto: uniffi_rusto_checksum_func_rusto_version: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_rusto_checksum_method_rusto_detect_text()
		})
		if checksum != 7671 {
			// If this happens try cleaning and rebuilding your project
			panic("rusto: uniffi_rusto_checksum_method_rusto_detect_text: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_rusto_checksum_method_rusto_detect_text_rgb()
		})
		if checksum != 14239 {
			// If this happens try cleaning and rebuilding your project
			panic("rusto: uniffi_rusto_checksum_method_rusto_detect_text_rgb: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_rusto_checksum_method_rusto_version()
		})
		if checksum != 27087 {
			// If this happens try cleaning and rebuilding your project
			panic("rusto: uniffi_rusto_checksum_method_rusto_version: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_rusto_checksum_constructor_rusto_initialize()
		})
		if checksum != 334 {
			// If this happens try cleaning and rebuilding your project
			panic("rusto: uniffi_rusto_checksum_constructor_rusto_initialize: UniFFI API checksum mismatch")
		}
	}
	{
		checksum := rustCall(func(_uniffiStatus *C.RustCallStatus) C.uint16_t {
			return C.uniffi_rusto_checksum_constructor_rusto_initialize_from_json()
		})
		if checksum != 23041 {
			// If this happens try cleaning and rebuilding your project
			panic("rusto: uniffi_rusto_checksum_constructor_rusto_initialize_from_json: UniFFI API checksum mismatch")
		}
	}
}

type FfiConverterUint32 struct{}

var FfiConverterUint32INSTANCE = FfiConverterUint32{}

func (FfiConverterUint32) Lower(value uint32) C.uint32_t {
	return C.uint32_t(value)
}

func (FfiConverterUint32) Write(writer io.Writer, value uint32) {
	writeUint32(writer, value)
}

func (FfiConverterUint32) Lift(value C.uint32_t) uint32 {
	return uint32(value)
}

func (FfiConverterUint32) Read(reader io.Reader) uint32 {
	return readUint32(reader)
}

type FfiDestroyerUint32 struct{}

func (FfiDestroyerUint32) Destroy(_ uint32) {}

type FfiConverterInt32 struct{}

var FfiConverterInt32INSTANCE = FfiConverterInt32{}

func (FfiConverterInt32) Lower(value int32) C.int32_t {
	return C.int32_t(value)
}

func (FfiConverterInt32) Write(writer io.Writer, value int32) {
	writeInt32(writer, value)
}

func (FfiConverterInt32) Lift(value C.int32_t) int32 {
	return int32(value)
}

func (FfiConverterInt32) Read(reader io.Reader) int32 {
	return readInt32(reader)
}

type FfiDestroyerInt32 struct{}

func (FfiDestroyerInt32) Destroy(_ int32) {}

type FfiConverterUint64 struct{}

var FfiConverterUint64INSTANCE = FfiConverterUint64{}

func (FfiConverterUint64) Lower(value uint64) C.uint64_t {
	return C.uint64_t(value)
}

func (FfiConverterUint64) Write(writer io.Writer, value uint64) {
	writeUint64(writer, value)
}

func (FfiConverterUint64) Lift(value C.uint64_t) uint64 {
	return uint64(value)
}

func (FfiConverterUint64) Read(reader io.Reader) uint64 {
	return readUint64(reader)
}

type FfiDestroyerUint64 struct{}

func (FfiDestroyerUint64) Destroy(_ uint64) {}

type FfiConverterFloat32 struct{}

var FfiConverterFloat32INSTANCE = FfiConverterFloat32{}

func (FfiConverterFloat32) Lower(value float32) C.float {
	return C.float(value)
}

func (FfiConverterFloat32) Write(writer io.Writer, value float32) {
	writeFloat32(writer, value)
}

func (FfiConverterFloat32) Lift(value C.float) float32 {
	return float32(value)
}

func (FfiConverterFloat32) Read(reader io.Reader) float32 {
	return readFloat32(reader)
}

type FfiDestroyerFloat32 struct{}

func (FfiDestroyerFloat32) Destroy(_ float32) {}

type FfiConverterBool struct{}

var FfiConverterBoolINSTANCE = FfiConverterBool{}

func (FfiConverterBool) Lower(value bool) C.int8_t {
	if value {
		return C.int8_t(1)
	}
	return C.int8_t(0)
}

func (FfiConverterBool) Write(writer io.Writer, value bool) {
	if value {
		writeInt8(writer, 1)
	} else {
		writeInt8(writer, 0)
	}
}

func (FfiConverterBool) Lift(value C.int8_t) bool {
	return value != 0
}

func (FfiConverterBool) Read(reader io.Reader) bool {
	return readInt8(reader) != 0
}

type FfiDestroyerBool struct{}

func (FfiDestroyerBool) Destroy(_ bool) {}

type FfiConverterString struct{}

var FfiConverterStringINSTANCE = FfiConverterString{}

func (FfiConverterString) Lift(rb RustBufferI) string {
	defer rb.Free()
	reader := rb.AsReader()
	b, err := io.ReadAll(reader)
	if err != nil {
		panic(fmt.Errorf("reading reader: %w", err))
	}
	return string(b)
}

func (FfiConverterString) Read(reader io.Reader) string {
	length := readInt32(reader)
	buffer := make([]byte, length)
	read_length, err := reader.Read(buffer)
	if err != nil && err != io.EOF {
		panic(err)
	}
	if read_length != int(length) {
		panic(fmt.Errorf("bad read length when reading string, expected %d, read %d", length, read_length))
	}
	return string(buffer)
}

func (FfiConverterString) Lower(value string) C.RustBuffer {
	return stringToRustBuffer(value)
}

func (c FfiConverterString) LowerExternal(value string) ExternalCRustBuffer {
	return RustBufferFromC(stringToRustBuffer(value))
}

func (FfiConverterString) Write(writer io.Writer, value string) {
	if len(value) > math.MaxInt32 {
		panic("String is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	write_length, err := io.WriteString(writer, value)
	if err != nil {
		panic(err)
	}
	if write_length != len(value) {
		panic(fmt.Errorf("bad write length when writing string, expected %d, written %d", len(value), write_length))
	}
}

type FfiDestroyerString struct{}

func (FfiDestroyerString) Destroy(_ string) {}

type FfiConverterBytes struct{}

var FfiConverterBytesINSTANCE = FfiConverterBytes{}

func (c FfiConverterBytes) Lower(value []byte) C.RustBuffer {
	return LowerIntoRustBuffer[[]byte](c, value)
}

func (c FfiConverterBytes) LowerExternal(value []byte) ExternalCRustBuffer {
	return RustBufferFromC(c.Lower(value))
}

func (c FfiConverterBytes) Write(writer io.Writer, value []byte) {
	if len(value) > math.MaxInt32 {
		panic("[]byte is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	write_length, err := writer.Write(value)
	if err != nil {
		panic(err)
	}
	if write_length != len(value) {
		panic(fmt.Errorf("bad write length when writing []byte, expected %d, written %d", len(value), write_length))
	}
}

func (c FfiConverterBytes) Lift(rb RustBufferI) []byte {
	return LiftFromRustBuffer[[]byte](c, rb)
}

func (c FfiConverterBytes) Read(reader io.Reader) []byte {
	length := readInt32(reader)
	buffer := make([]byte, length)
	read_length, err := reader.Read(buffer)
	if err != nil && err != io.EOF {
		panic(err)
	}
	if read_length != int(length) {
		panic(fmt.Errorf("bad read length when reading []byte, expected %d, read %d", length, read_length))
	}
	return buffer
}

type FfiDestroyerBytes struct{}

func (FfiDestroyerBytes) Destroy(_ []byte) {}

// Below is an implementation of synchronization requirements outlined in the link.
// https://github.com/mozilla/uniffi-rs/blob/0dc031132d9493ca812c3af6e7dd60ad2ea95bf0/uniffi_bindgen/src/bindings/kotlin/templates/ObjectRuntime.kt#L31

type FfiObject struct {
	handle        C.uint64_t
	callCounter   atomic.Int64
	cloneFunction func(C.uint64_t, *C.RustCallStatus) C.uint64_t
	freeFunction  func(C.uint64_t, *C.RustCallStatus)
	destroyed     atomic.Bool
}

func newFfiObject(
	handle C.uint64_t,
	cloneFunction func(C.uint64_t, *C.RustCallStatus) C.uint64_t,
	freeFunction func(C.uint64_t, *C.RustCallStatus),
) FfiObject {
	return FfiObject{
		handle:        handle,
		cloneFunction: cloneFunction,
		freeFunction:  freeFunction,
	}
}

func (ffiObject *FfiObject) incrementPointer(debugName string) C.uint64_t {
	for {
		counter := ffiObject.callCounter.Load()
		if counter <= -1 {
			panic(fmt.Errorf("%v object has already been destroyed", debugName))
		}
		if counter == math.MaxInt64 {
			panic(fmt.Errorf("%v object call counter would overflow", debugName))
		}
		if ffiObject.callCounter.CompareAndSwap(counter, counter+1) {
			break
		}
	}

	return rustCall(func(status *C.RustCallStatus) C.uint64_t {
		return ffiObject.cloneFunction(ffiObject.handle, status)
	})
}

func (ffiObject *FfiObject) decrementPointer() {
	if ffiObject.callCounter.Add(-1) == -1 {
		ffiObject.freeRustArcPtr()
	}
}

func (ffiObject *FfiObject) destroy() {
	if ffiObject.destroyed.CompareAndSwap(false, true) {
		if ffiObject.callCounter.Add(-1) == -1 {
			ffiObject.freeRustArcPtr()
		}
	}
}

func (ffiObject *FfiObject) freeRustArcPtr() {
	if ffiObject.handle == 0 {
		return
	}
	rustCall(func(status *C.RustCallStatus) int32 {
		ffiObject.freeFunction(ffiObject.handle, status)
		return 0
	})
}

type RustOInterface interface {
	DetectText(source ImageSource, options OcrRunOptions) (DetectTextResult, error)
	DetectTextRgb(pixels []byte, width uint32, height uint32, stride uint64, options OcrRunOptions) (DetectTextResult, error)
	Version() string
}
type RustO struct {
	ffiObject FfiObject
}

func RustOInitialize(config InitializeConfig) (*RustO, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*RustoBindingError](FfiConverterRustoBindingError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_rusto_fn_constructor_rusto_initialize(FfiConverterInitializeConfigINSTANCE.Lower(config), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *RustO
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterRustOINSTANCE.Lift(_uniffiRV), nil
	}
}

func RustOInitializeFromJson(configJson string) (*RustO, error) {
	_uniffiRV, _uniffiErr := rustCallWithError[*RustoBindingError](FfiConverterRustoBindingError{}, func(_uniffiStatus *C.RustCallStatus) C.uint64_t {
		return C.uniffi_rusto_fn_constructor_rusto_initialize_from_json(FfiConverterStringINSTANCE.Lower(configJson), _uniffiStatus)
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue *RustO
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterRustOINSTANCE.Lift(_uniffiRV), nil
	}
}

func (_self *RustO) DetectText(source ImageSource, options OcrRunOptions) (DetectTextResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*RustO")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*RustoBindingError](FfiConverterRustoBindingError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_rusto_fn_method_rusto_detect_text(
				_pointer, FfiConverterImageSourceINSTANCE.Lower(source), FfiConverterOcrRunOptionsINSTANCE.Lower(options), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue DetectTextResult
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterDetectTextResultINSTANCE.Lift(_uniffiRV), nil
	}
}

func (_self *RustO) DetectTextRgb(pixels []byte, width uint32, height uint32, stride uint64, options OcrRunOptions) (DetectTextResult, error) {
	_pointer := _self.ffiObject.incrementPointer("*RustO")
	defer _self.ffiObject.decrementPointer()
	_uniffiRV, _uniffiErr := rustCallWithError[*RustoBindingError](FfiConverterRustoBindingError{}, func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_rusto_fn_method_rusto_detect_text_rgb(
				_pointer, FfiConverterBytesINSTANCE.Lower(pixels), FfiConverterUint32INSTANCE.Lower(width), FfiConverterUint32INSTANCE.Lower(height), FfiConverterUint64INSTANCE.Lower(stride), FfiConverterOcrRunOptionsINSTANCE.Lower(options), _uniffiStatus),
		}
	})
	if _uniffiErr != nil {
		var _uniffiDefaultValue DetectTextResult
		return _uniffiDefaultValue, _uniffiErr
	} else {
		return FfiConverterDetectTextResultINSTANCE.Lift(_uniffiRV), nil
	}
}

func (_self *RustO) Version() string {
	_pointer := _self.ffiObject.incrementPointer("*RustO")
	defer _self.ffiObject.decrementPointer()
	return FfiConverterStringINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_rusto_fn_method_rusto_version(
				_pointer, _uniffiStatus),
		}
	}))
}
func (object *RustO) Destroy() {
	runtime.SetFinalizer(object, nil)
	object.ffiObject.destroy()
}

type FfiConverterRustO struct{}

var FfiConverterRustOINSTANCE = FfiConverterRustO{}

func (c FfiConverterRustO) Lift(handle C.uint64_t) *RustO {
	result := &RustO{
		newFfiObject(
			handle,
			func(handle C.uint64_t, status *C.RustCallStatus) C.uint64_t {
				return C.uniffi_rusto_fn_clone_rusto(handle, status)
			},
			func(handle C.uint64_t, status *C.RustCallStatus) {
				C.uniffi_rusto_fn_free_rusto(handle, status)
			},
		),
	}
	runtime.SetFinalizer(result, (*RustO).Destroy)
	return result
}

func (c FfiConverterRustO) Read(reader io.Reader) *RustO {
	return c.Lift(C.uint64_t(readUint64(reader)))
}

func (c FfiConverterRustO) Lower(value *RustO) C.uint64_t {
	// TODO: this is bad - all synchronization from ObjectRuntime.go is discarded here,
	// because the handle will be decremented immediately after this function returns,
	// and someone will be left holding onto a non-locked handle.
	handle := value.ffiObject.incrementPointer("*RustO")
	defer value.ffiObject.decrementPointer()
	return handle
}

func (c FfiConverterRustO) Write(writer io.Writer, value *RustO) {
	writeUint64(writer, uint64(c.Lower(value)))
}

func LiftFromExternalRustO(handle uint64) *RustO {
	return FfiConverterRustOINSTANCE.Lift(C.uint64_t(handle))
}

func LowerToExternalRustO(value *RustO) uint64 {
	return uint64(FfiConverterRustOINSTANCE.Lower(value))
}

type FfiDestroyerRustO struct{}

func (_ FfiDestroyerRustO) Destroy(value *RustO) {
	value.Destroy()
}

type CalibrationOptions struct {
	DescreenStrength *float32
}

func (r *CalibrationOptions) Destroy() {
	FfiDestroyerOptionalFloat32{}.Destroy(r.DescreenStrength)
}

type FfiConverterCalibrationOptions struct{}

var FfiConverterCalibrationOptionsINSTANCE = FfiConverterCalibrationOptions{}

func (c FfiConverterCalibrationOptions) Lift(rb RustBufferI) CalibrationOptions {
	return LiftFromRustBuffer[CalibrationOptions](c, rb)
}

func (c FfiConverterCalibrationOptions) Read(reader io.Reader) CalibrationOptions {
	return CalibrationOptions{
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
	}
}

func (c FfiConverterCalibrationOptions) Lower(value CalibrationOptions) C.RustBuffer {
	return LowerIntoRustBuffer[CalibrationOptions](c, value)
}

func (c FfiConverterCalibrationOptions) LowerExternal(value CalibrationOptions) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[CalibrationOptions](c, value))
}

func (c FfiConverterCalibrationOptions) Write(writer io.Writer, value CalibrationOptions) {
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.DescreenStrength)
}

type FfiDestroyerCalibrationOptions struct{}

func (_ FfiDestroyerCalibrationOptions) Destroy(value CalibrationOptions) {
	value.Destroy()
}

type ClassificationConfig struct {
	Enabled   *bool
	ModelPath *string
	Thresh    *float32
}

func (r *ClassificationConfig) Destroy() {
	FfiDestroyerOptionalBool{}.Destroy(r.Enabled)
	FfiDestroyerOptionalString{}.Destroy(r.ModelPath)
	FfiDestroyerOptionalFloat32{}.Destroy(r.Thresh)
}

type FfiConverterClassificationConfig struct{}

var FfiConverterClassificationConfigINSTANCE = FfiConverterClassificationConfig{}

func (c FfiConverterClassificationConfig) Lift(rb RustBufferI) ClassificationConfig {
	return LiftFromRustBuffer[ClassificationConfig](c, rb)
}

func (c FfiConverterClassificationConfig) Read(reader io.Reader) ClassificationConfig {
	return ClassificationConfig{
		FfiConverterOptionalBoolINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
	}
}

func (c FfiConverterClassificationConfig) Lower(value ClassificationConfig) C.RustBuffer {
	return LowerIntoRustBuffer[ClassificationConfig](c, value)
}

func (c FfiConverterClassificationConfig) LowerExternal(value ClassificationConfig) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ClassificationConfig](c, value))
}

func (c FfiConverterClassificationConfig) Write(writer io.Writer, value ClassificationConfig) {
	FfiConverterOptionalBoolINSTANCE.Write(writer, value.Enabled)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.ModelPath)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.Thresh)
}

type FfiDestroyerClassificationConfig struct{}

func (_ FfiDestroyerClassificationConfig) Destroy(value ClassificationConfig) {
	value.Destroy()
}

type DetectionConfig struct {
	Enabled      *bool
	ModelPath    *string
	Thresh       *float32
	BoxThresh    *float32
	UnclipRatio  *float32
	LimitSideLen *int32
	LimitType    *string
	UseDilation  *bool
}

func (r *DetectionConfig) Destroy() {
	FfiDestroyerOptionalBool{}.Destroy(r.Enabled)
	FfiDestroyerOptionalString{}.Destroy(r.ModelPath)
	FfiDestroyerOptionalFloat32{}.Destroy(r.Thresh)
	FfiDestroyerOptionalFloat32{}.Destroy(r.BoxThresh)
	FfiDestroyerOptionalFloat32{}.Destroy(r.UnclipRatio)
	FfiDestroyerOptionalInt32{}.Destroy(r.LimitSideLen)
	FfiDestroyerOptionalString{}.Destroy(r.LimitType)
	FfiDestroyerOptionalBool{}.Destroy(r.UseDilation)
}

type FfiConverterDetectionConfig struct{}

var FfiConverterDetectionConfigINSTANCE = FfiConverterDetectionConfig{}

func (c FfiConverterDetectionConfig) Lift(rb RustBufferI) DetectionConfig {
	return LiftFromRustBuffer[DetectionConfig](c, rb)
}

func (c FfiConverterDetectionConfig) Read(reader io.Reader) DetectionConfig {
	return DetectionConfig{
		FfiConverterOptionalBoolINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalInt32INSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalBoolINSTANCE.Read(reader),
	}
}

func (c FfiConverterDetectionConfig) Lower(value DetectionConfig) C.RustBuffer {
	return LowerIntoRustBuffer[DetectionConfig](c, value)
}

func (c FfiConverterDetectionConfig) LowerExternal(value DetectionConfig) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[DetectionConfig](c, value))
}

func (c FfiConverterDetectionConfig) Write(writer io.Writer, value DetectionConfig) {
	FfiConverterOptionalBoolINSTANCE.Write(writer, value.Enabled)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.ModelPath)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.Thresh)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.BoxThresh)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.UnclipRatio)
	FfiConverterOptionalInt32INSTANCE.Write(writer, value.LimitSideLen)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.LimitType)
	FfiConverterOptionalBoolINSTANCE.Write(writer, value.UseDilation)
}

type FfiDestroyerDetectionConfig struct{}

func (_ FfiDestroyerDetectionConfig) Destroy(value DetectionConfig) {
	value.Destroy()
}

type DetectionRunOptions struct {
	LimitSideLen *int32
	LimitType    *string
	Mean         *[]float32
	Std          *[]float32
	Postprocess  *PostprocessRunOptions
}

func (r *DetectionRunOptions) Destroy() {
	FfiDestroyerOptionalInt32{}.Destroy(r.LimitSideLen)
	FfiDestroyerOptionalString{}.Destroy(r.LimitType)
	FfiDestroyerOptionalSequenceFloat32{}.Destroy(r.Mean)
	FfiDestroyerOptionalSequenceFloat32{}.Destroy(r.Std)
	FfiDestroyerOptionalPostprocessRunOptions{}.Destroy(r.Postprocess)
}

type FfiConverterDetectionRunOptions struct{}

var FfiConverterDetectionRunOptionsINSTANCE = FfiConverterDetectionRunOptions{}

func (c FfiConverterDetectionRunOptions) Lift(rb RustBufferI) DetectionRunOptions {
	return LiftFromRustBuffer[DetectionRunOptions](c, rb)
}

func (c FfiConverterDetectionRunOptions) Read(reader io.Reader) DetectionRunOptions {
	return DetectionRunOptions{
		FfiConverterOptionalInt32INSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalSequenceFloat32INSTANCE.Read(reader),
		FfiConverterOptionalSequenceFloat32INSTANCE.Read(reader),
		FfiConverterOptionalPostprocessRunOptionsINSTANCE.Read(reader),
	}
}

func (c FfiConverterDetectionRunOptions) Lower(value DetectionRunOptions) C.RustBuffer {
	return LowerIntoRustBuffer[DetectionRunOptions](c, value)
}

func (c FfiConverterDetectionRunOptions) LowerExternal(value DetectionRunOptions) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[DetectionRunOptions](c, value))
}

func (c FfiConverterDetectionRunOptions) Write(writer io.Writer, value DetectionRunOptions) {
	FfiConverterOptionalInt32INSTANCE.Write(writer, value.LimitSideLen)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.LimitType)
	FfiConverterOptionalSequenceFloat32INSTANCE.Write(writer, value.Mean)
	FfiConverterOptionalSequenceFloat32INSTANCE.Write(writer, value.Std)
	FfiConverterOptionalPostprocessRunOptionsINSTANCE.Write(writer, value.Postprocess)
}

type FfiDestroyerDetectionRunOptions struct{}

func (_ FfiDestroyerDetectionRunOptions) Destroy(value DetectionRunOptions) {
	value.Destroy()
}

type Frame struct {
	Width  float32
	Height float32
	Top    float32
	Left   float32
}

func (r *Frame) Destroy() {
	FfiDestroyerFloat32{}.Destroy(r.Width)
	FfiDestroyerFloat32{}.Destroy(r.Height)
	FfiDestroyerFloat32{}.Destroy(r.Top)
	FfiDestroyerFloat32{}.Destroy(r.Left)
}

type FfiConverterFrame struct{}

var FfiConverterFrameINSTANCE = FfiConverterFrame{}

func (c FfiConverterFrame) Lift(rb RustBufferI) Frame {
	return LiftFromRustBuffer[Frame](c, rb)
}

func (c FfiConverterFrame) Read(reader io.Reader) Frame {
	return Frame{
		FfiConverterFloat32INSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
	}
}

func (c FfiConverterFrame) Lower(value Frame) C.RustBuffer {
	return LowerIntoRustBuffer[Frame](c, value)
}

func (c FfiConverterFrame) LowerExternal(value Frame) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[Frame](c, value))
}

func (c FfiConverterFrame) Write(writer io.Writer, value Frame) {
	FfiConverterFloat32INSTANCE.Write(writer, value.Width)
	FfiConverterFloat32INSTANCE.Write(writer, value.Height)
	FfiConverterFloat32INSTANCE.Write(writer, value.Top)
	FfiConverterFloat32INSTANCE.Write(writer, value.Left)
}

type FfiDestroyerFrame struct{}

func (_ FfiDestroyerFrame) Destroy(value Frame) {
	value.Destroy()
}

type InitializeConfig struct {
	Template       *string
	Detection      *DetectionConfig
	Recognition    *RecognitionConfig
	Classification *ClassificationConfig
	Orientation    *OrientationConfig
	Unwarp         *UnwarpConfig
}

func (r *InitializeConfig) Destroy() {
	FfiDestroyerOptionalString{}.Destroy(r.Template)
	FfiDestroyerOptionalDetectionConfig{}.Destroy(r.Detection)
	FfiDestroyerOptionalRecognitionConfig{}.Destroy(r.Recognition)
	FfiDestroyerOptionalClassificationConfig{}.Destroy(r.Classification)
	FfiDestroyerOptionalOrientationConfig{}.Destroy(r.Orientation)
	FfiDestroyerOptionalUnwarpConfig{}.Destroy(r.Unwarp)
}

type FfiConverterInitializeConfig struct{}

var FfiConverterInitializeConfigINSTANCE = FfiConverterInitializeConfig{}

func (c FfiConverterInitializeConfig) Lift(rb RustBufferI) InitializeConfig {
	return LiftFromRustBuffer[InitializeConfig](c, rb)
}

func (c FfiConverterInitializeConfig) Read(reader io.Reader) InitializeConfig {
	return InitializeConfig{
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalDetectionConfigINSTANCE.Read(reader),
		FfiConverterOptionalRecognitionConfigINSTANCE.Read(reader),
		FfiConverterOptionalClassificationConfigINSTANCE.Read(reader),
		FfiConverterOptionalOrientationConfigINSTANCE.Read(reader),
		FfiConverterOptionalUnwarpConfigINSTANCE.Read(reader),
	}
}

func (c FfiConverterInitializeConfig) Lower(value InitializeConfig) C.RustBuffer {
	return LowerIntoRustBuffer[InitializeConfig](c, value)
}

func (c FfiConverterInitializeConfig) LowerExternal(value InitializeConfig) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[InitializeConfig](c, value))
}

func (c FfiConverterInitializeConfig) Write(writer io.Writer, value InitializeConfig) {
	FfiConverterOptionalStringINSTANCE.Write(writer, value.Template)
	FfiConverterOptionalDetectionConfigINSTANCE.Write(writer, value.Detection)
	FfiConverterOptionalRecognitionConfigINSTANCE.Write(writer, value.Recognition)
	FfiConverterOptionalClassificationConfigINSTANCE.Write(writer, value.Classification)
	FfiConverterOptionalOrientationConfigINSTANCE.Write(writer, value.Orientation)
	FfiConverterOptionalUnwarpConfigINSTANCE.Write(writer, value.Unwarp)
}

type FfiDestroyerInitializeConfig struct{}

func (_ FfiDestroyerInitializeConfig) Destroy(value InitializeConfig) {
	value.Destroy()
}

type OcrRunOptions struct {
	Output           OutputGranularity
	LineYThreshold   *float32
	WordXThreshold   *float32
	TextScore        *float32
	Classification   *bool
	Orientation      *bool
	MinHeight        *float32
	MaxSideLen       *float32
	MinSideLen       *float32
	WidthHeightRatio *float32
	Detection        *DetectionRunOptions
	Postprocess      *PostprocessRunOptions
	Calibration      *CalibrationOptions
	Optimization     *OptimizationOptions
}

func (r *OcrRunOptions) Destroy() {
	FfiDestroyerOutputGranularity{}.Destroy(r.Output)
	FfiDestroyerOptionalFloat32{}.Destroy(r.LineYThreshold)
	FfiDestroyerOptionalFloat32{}.Destroy(r.WordXThreshold)
	FfiDestroyerOptionalFloat32{}.Destroy(r.TextScore)
	FfiDestroyerOptionalBool{}.Destroy(r.Classification)
	FfiDestroyerOptionalBool{}.Destroy(r.Orientation)
	FfiDestroyerOptionalFloat32{}.Destroy(r.MinHeight)
	FfiDestroyerOptionalFloat32{}.Destroy(r.MaxSideLen)
	FfiDestroyerOptionalFloat32{}.Destroy(r.MinSideLen)
	FfiDestroyerOptionalFloat32{}.Destroy(r.WidthHeightRatio)
	FfiDestroyerOptionalDetectionRunOptions{}.Destroy(r.Detection)
	FfiDestroyerOptionalPostprocessRunOptions{}.Destroy(r.Postprocess)
	FfiDestroyerOptionalCalibrationOptions{}.Destroy(r.Calibration)
	FfiDestroyerOptionalOptimizationOptions{}.Destroy(r.Optimization)
}

type FfiConverterOcrRunOptions struct{}

var FfiConverterOcrRunOptionsINSTANCE = FfiConverterOcrRunOptions{}

func (c FfiConverterOcrRunOptions) Lift(rb RustBufferI) OcrRunOptions {
	return LiftFromRustBuffer[OcrRunOptions](c, rb)
}

func (c FfiConverterOcrRunOptions) Read(reader io.Reader) OcrRunOptions {
	return OcrRunOptions{
		FfiConverterOutputGranularityINSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalBoolINSTANCE.Read(reader),
		FfiConverterOptionalBoolINSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalDetectionRunOptionsINSTANCE.Read(reader),
		FfiConverterOptionalPostprocessRunOptionsINSTANCE.Read(reader),
		FfiConverterOptionalCalibrationOptionsINSTANCE.Read(reader),
		FfiConverterOptionalOptimizationOptionsINSTANCE.Read(reader),
	}
}

func (c FfiConverterOcrRunOptions) Lower(value OcrRunOptions) C.RustBuffer {
	return LowerIntoRustBuffer[OcrRunOptions](c, value)
}

func (c FfiConverterOcrRunOptions) LowerExternal(value OcrRunOptions) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[OcrRunOptions](c, value))
}

func (c FfiConverterOcrRunOptions) Write(writer io.Writer, value OcrRunOptions) {
	FfiConverterOutputGranularityINSTANCE.Write(writer, value.Output)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.LineYThreshold)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.WordXThreshold)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.TextScore)
	FfiConverterOptionalBoolINSTANCE.Write(writer, value.Classification)
	FfiConverterOptionalBoolINSTANCE.Write(writer, value.Orientation)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.MinHeight)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.MaxSideLen)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.MinSideLen)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.WidthHeightRatio)
	FfiConverterOptionalDetectionRunOptionsINSTANCE.Write(writer, value.Detection)
	FfiConverterOptionalPostprocessRunOptionsINSTANCE.Write(writer, value.Postprocess)
	FfiConverterOptionalCalibrationOptionsINSTANCE.Write(writer, value.Calibration)
	FfiConverterOptionalOptimizationOptionsINSTANCE.Write(writer, value.Optimization)
}

type FfiDestroyerOcrRunOptions struct{}

func (_ FfiDestroyerOcrRunOptions) Destroy(value OcrRunOptions) {
	value.Destroy()
}

type OptimizationOptions struct {
	CropPaddingX *float32
	CropPaddingY *float32
}

func (r *OptimizationOptions) Destroy() {
	FfiDestroyerOptionalFloat32{}.Destroy(r.CropPaddingX)
	FfiDestroyerOptionalFloat32{}.Destroy(r.CropPaddingY)
}

type FfiConverterOptimizationOptions struct{}

var FfiConverterOptimizationOptionsINSTANCE = FfiConverterOptimizationOptions{}

func (c FfiConverterOptimizationOptions) Lift(rb RustBufferI) OptimizationOptions {
	return LiftFromRustBuffer[OptimizationOptions](c, rb)
}

func (c FfiConverterOptimizationOptions) Read(reader io.Reader) OptimizationOptions {
	return OptimizationOptions{
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
	}
}

func (c FfiConverterOptimizationOptions) Lower(value OptimizationOptions) C.RustBuffer {
	return LowerIntoRustBuffer[OptimizationOptions](c, value)
}

func (c FfiConverterOptimizationOptions) LowerExternal(value OptimizationOptions) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[OptimizationOptions](c, value))
}

func (c FfiConverterOptimizationOptions) Write(writer io.Writer, value OptimizationOptions) {
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.CropPaddingX)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.CropPaddingY)
}

type FfiDestroyerOptimizationOptions struct{}

func (_ FfiDestroyerOptimizationOptions) Destroy(value OptimizationOptions) {
	value.Destroy()
}

type OrientationConfig struct {
	Enabled   *bool
	ModelPath *string
	Thresh    *float32
}

func (r *OrientationConfig) Destroy() {
	FfiDestroyerOptionalBool{}.Destroy(r.Enabled)
	FfiDestroyerOptionalString{}.Destroy(r.ModelPath)
	FfiDestroyerOptionalFloat32{}.Destroy(r.Thresh)
}

type FfiConverterOrientationConfig struct{}

var FfiConverterOrientationConfigINSTANCE = FfiConverterOrientationConfig{}

func (c FfiConverterOrientationConfig) Lift(rb RustBufferI) OrientationConfig {
	return LiftFromRustBuffer[OrientationConfig](c, rb)
}

func (c FfiConverterOrientationConfig) Read(reader io.Reader) OrientationConfig {
	return OrientationConfig{
		FfiConverterOptionalBoolINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
	}
}

func (c FfiConverterOrientationConfig) Lower(value OrientationConfig) C.RustBuffer {
	return LowerIntoRustBuffer[OrientationConfig](c, value)
}

func (c FfiConverterOrientationConfig) LowerExternal(value OrientationConfig) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[OrientationConfig](c, value))
}

func (c FfiConverterOrientationConfig) Write(writer io.Writer, value OrientationConfig) {
	FfiConverterOptionalBoolINSTANCE.Write(writer, value.Enabled)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.ModelPath)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.Thresh)
}

type FfiDestroyerOrientationConfig struct{}

func (_ FfiDestroyerOrientationConfig) Destroy(value OrientationConfig) {
	value.Destroy()
}

type Point2D struct {
	X float32
	Y float32
}

func (r *Point2D) Destroy() {
	FfiDestroyerFloat32{}.Destroy(r.X)
	FfiDestroyerFloat32{}.Destroy(r.Y)
}

type FfiConverterPoint2D struct{}

var FfiConverterPoint2DINSTANCE = FfiConverterPoint2D{}

func (c FfiConverterPoint2D) Lift(rb RustBufferI) Point2D {
	return LiftFromRustBuffer[Point2D](c, rb)
}

func (c FfiConverterPoint2D) Read(reader io.Reader) Point2D {
	return Point2D{
		FfiConverterFloat32INSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
	}
}

func (c FfiConverterPoint2D) Lower(value Point2D) C.RustBuffer {
	return LowerIntoRustBuffer[Point2D](c, value)
}

func (c FfiConverterPoint2D) LowerExternal(value Point2D) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[Point2D](c, value))
}

func (c FfiConverterPoint2D) Write(writer io.Writer, value Point2D) {
	FfiConverterFloat32INSTANCE.Write(writer, value.X)
	FfiConverterFloat32INSTANCE.Write(writer, value.Y)
}

type FfiDestroyerPoint2D struct{}

func (_ FfiDestroyerPoint2D) Destroy(value Point2D) {
	value.Destroy()
}

type PostprocessRunOptions struct {
	Threshold     *float32
	BoxThreshold  *float32
	MaxCandidates *int32
	UnclipRatio   *float32
	UseDilation   *bool
}

func (r *PostprocessRunOptions) Destroy() {
	FfiDestroyerOptionalFloat32{}.Destroy(r.Threshold)
	FfiDestroyerOptionalFloat32{}.Destroy(r.BoxThreshold)
	FfiDestroyerOptionalInt32{}.Destroy(r.MaxCandidates)
	FfiDestroyerOptionalFloat32{}.Destroy(r.UnclipRatio)
	FfiDestroyerOptionalBool{}.Destroy(r.UseDilation)
}

type FfiConverterPostprocessRunOptions struct{}

var FfiConverterPostprocessRunOptionsINSTANCE = FfiConverterPostprocessRunOptions{}

func (c FfiConverterPostprocessRunOptions) Lift(rb RustBufferI) PostprocessRunOptions {
	return LiftFromRustBuffer[PostprocessRunOptions](c, rb)
}

func (c FfiConverterPostprocessRunOptions) Read(reader io.Reader) PostprocessRunOptions {
	return PostprocessRunOptions{
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalInt32INSTANCE.Read(reader),
		FfiConverterOptionalFloat32INSTANCE.Read(reader),
		FfiConverterOptionalBoolINSTANCE.Read(reader),
	}
}

func (c FfiConverterPostprocessRunOptions) Lower(value PostprocessRunOptions) C.RustBuffer {
	return LowerIntoRustBuffer[PostprocessRunOptions](c, value)
}

func (c FfiConverterPostprocessRunOptions) LowerExternal(value PostprocessRunOptions) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[PostprocessRunOptions](c, value))
}

func (c FfiConverterPostprocessRunOptions) Write(writer io.Writer, value PostprocessRunOptions) {
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.Threshold)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.BoxThreshold)
	FfiConverterOptionalInt32INSTANCE.Write(writer, value.MaxCandidates)
	FfiConverterOptionalFloat32INSTANCE.Write(writer, value.UnclipRatio)
	FfiConverterOptionalBoolINSTANCE.Write(writer, value.UseDilation)
}

type FfiDestroyerPostprocessRunOptions struct{}

func (_ FfiDestroyerPostprocessRunOptions) Destroy(value PostprocessRunOptions) {
	value.Destroy()
}

type RecognitionConfig struct {
	Enabled   *bool
	ModelPath *string
	DictPath  *string
}

func (r *RecognitionConfig) Destroy() {
	FfiDestroyerOptionalBool{}.Destroy(r.Enabled)
	FfiDestroyerOptionalString{}.Destroy(r.ModelPath)
	FfiDestroyerOptionalString{}.Destroy(r.DictPath)
}

type FfiConverterRecognitionConfig struct{}

var FfiConverterRecognitionConfigINSTANCE = FfiConverterRecognitionConfig{}

func (c FfiConverterRecognitionConfig) Lift(rb RustBufferI) RecognitionConfig {
	return LiftFromRustBuffer[RecognitionConfig](c, rb)
}

func (c FfiConverterRecognitionConfig) Read(reader io.Reader) RecognitionConfig {
	return RecognitionConfig{
		FfiConverterOptionalBoolINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterRecognitionConfig) Lower(value RecognitionConfig) C.RustBuffer {
	return LowerIntoRustBuffer[RecognitionConfig](c, value)
}

func (c FfiConverterRecognitionConfig) LowerExternal(value RecognitionConfig) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[RecognitionConfig](c, value))
}

func (c FfiConverterRecognitionConfig) Write(writer io.Writer, value RecognitionConfig) {
	FfiConverterOptionalBoolINSTANCE.Write(writer, value.Enabled)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.ModelPath)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.DictPath)
}

type FfiDestroyerRecognitionConfig struct{}

func (_ FfiDestroyerRecognitionConfig) Destroy(value RecognitionConfig) {
	value.Destroy()
}

type TextResult struct {
	Text      string
	Score     float32
	BoxPoints []Point2D
	Frame     Frame
}

func (r *TextResult) Destroy() {
	FfiDestroyerString{}.Destroy(r.Text)
	FfiDestroyerFloat32{}.Destroy(r.Score)
	FfiDestroyerSequencePoint2D{}.Destroy(r.BoxPoints)
	FfiDestroyerFrame{}.Destroy(r.Frame)
}

type FfiConverterTextResult struct{}

var FfiConverterTextResultINSTANCE = FfiConverterTextResult{}

func (c FfiConverterTextResult) Lift(rb RustBufferI) TextResult {
	return LiftFromRustBuffer[TextResult](c, rb)
}

func (c FfiConverterTextResult) Read(reader io.Reader) TextResult {
	return TextResult{
		FfiConverterStringINSTANCE.Read(reader),
		FfiConverterFloat32INSTANCE.Read(reader),
		FfiConverterSequencePoint2DINSTANCE.Read(reader),
		FfiConverterFrameINSTANCE.Read(reader),
	}
}

func (c FfiConverterTextResult) Lower(value TextResult) C.RustBuffer {
	return LowerIntoRustBuffer[TextResult](c, value)
}

func (c FfiConverterTextResult) LowerExternal(value TextResult) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[TextResult](c, value))
}

func (c FfiConverterTextResult) Write(writer io.Writer, value TextResult) {
	FfiConverterStringINSTANCE.Write(writer, value.Text)
	FfiConverterFloat32INSTANCE.Write(writer, value.Score)
	FfiConverterSequencePoint2DINSTANCE.Write(writer, value.BoxPoints)
	FfiConverterFrameINSTANCE.Write(writer, value.Frame)
}

type FfiDestroyerTextResult struct{}

func (_ FfiDestroyerTextResult) Destroy(value TextResult) {
	value.Destroy()
}

type UnwarpConfig struct {
	Enabled   *bool
	ModelPath *string
}

func (r *UnwarpConfig) Destroy() {
	FfiDestroyerOptionalBool{}.Destroy(r.Enabled)
	FfiDestroyerOptionalString{}.Destroy(r.ModelPath)
}

type FfiConverterUnwarpConfig struct{}

var FfiConverterUnwarpConfigINSTANCE = FfiConverterUnwarpConfig{}

func (c FfiConverterUnwarpConfig) Lift(rb RustBufferI) UnwarpConfig {
	return LiftFromRustBuffer[UnwarpConfig](c, rb)
}

func (c FfiConverterUnwarpConfig) Read(reader io.Reader) UnwarpConfig {
	return UnwarpConfig{
		FfiConverterOptionalBoolINSTANCE.Read(reader),
		FfiConverterOptionalStringINSTANCE.Read(reader),
	}
}

func (c FfiConverterUnwarpConfig) Lower(value UnwarpConfig) C.RustBuffer {
	return LowerIntoRustBuffer[UnwarpConfig](c, value)
}

func (c FfiConverterUnwarpConfig) LowerExternal(value UnwarpConfig) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[UnwarpConfig](c, value))
}

func (c FfiConverterUnwarpConfig) Write(writer io.Writer, value UnwarpConfig) {
	FfiConverterOptionalBoolINSTANCE.Write(writer, value.Enabled)
	FfiConverterOptionalStringINSTANCE.Write(writer, value.ModelPath)
}

type FfiDestroyerUnwarpConfig struct{}

func (_ FfiDestroyerUnwarpConfig) Destroy(value UnwarpConfig) {
	value.Destroy()
}

type DetectTextResult interface {
	Destroy()
}
type DetectTextResultStructured struct {
	Results []TextResult
}

func (e DetectTextResultStructured) Destroy() {
	FfiDestroyerSequenceTextResult{}.Destroy(e.Results)
}

type DetectTextResultSpatial struct {
	Text string
}

func (e DetectTextResultSpatial) Destroy() {
	FfiDestroyerString{}.Destroy(e.Text)
}

type FfiConverterDetectTextResult struct{}

var FfiConverterDetectTextResultINSTANCE = FfiConverterDetectTextResult{}

func (c FfiConverterDetectTextResult) Lift(rb RustBufferI) DetectTextResult {
	return LiftFromRustBuffer[DetectTextResult](c, rb)
}

func (c FfiConverterDetectTextResult) Lower(value DetectTextResult) C.RustBuffer {
	return LowerIntoRustBuffer[DetectTextResult](c, value)
}

func (c FfiConverterDetectTextResult) LowerExternal(value DetectTextResult) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[DetectTextResult](c, value))
}
func (FfiConverterDetectTextResult) Read(reader io.Reader) DetectTextResult {
	id := readInt32(reader)
	switch id {
	case 1:
		return DetectTextResultStructured{
			FfiConverterSequenceTextResultINSTANCE.Read(reader),
		}
	case 2:
		return DetectTextResultSpatial{
			FfiConverterStringINSTANCE.Read(reader),
		}
	default:
		panic(fmt.Sprintf("invalid enum value %v in FfiConverterDetectTextResult.Read()", id))
	}
}

func (FfiConverterDetectTextResult) Write(writer io.Writer, value DetectTextResult) {
	switch variant_value := value.(type) {
	case DetectTextResultStructured:
		writeInt32(writer, 1)
		FfiConverterSequenceTextResultINSTANCE.Write(writer, variant_value.Results)
	case DetectTextResultSpatial:
		writeInt32(writer, 2)
		FfiConverterStringINSTANCE.Write(writer, variant_value.Text)
	default:
		_ = variant_value
		panic(fmt.Sprintf("invalid enum value `%v` in FfiConverterDetectTextResult.Write", value))
	}
}

type FfiDestroyerDetectTextResult struct{}

func (_ FfiDestroyerDetectTextResult) Destroy(value DetectTextResult) {
	value.Destroy()
}

type ImageSource interface {
	Destroy()
}
type ImageSourcePath struct {
	Path string
}

func (e ImageSourcePath) Destroy() {
	FfiDestroyerString{}.Destroy(e.Path)
}

type ImageSourceBytes struct {
	Data []byte
}

func (e ImageSourceBytes) Destroy() {
	FfiDestroyerBytes{}.Destroy(e.Data)
}

type FfiConverterImageSource struct{}

var FfiConverterImageSourceINSTANCE = FfiConverterImageSource{}

func (c FfiConverterImageSource) Lift(rb RustBufferI) ImageSource {
	return LiftFromRustBuffer[ImageSource](c, rb)
}

func (c FfiConverterImageSource) Lower(value ImageSource) C.RustBuffer {
	return LowerIntoRustBuffer[ImageSource](c, value)
}

func (c FfiConverterImageSource) LowerExternal(value ImageSource) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[ImageSource](c, value))
}
func (FfiConverterImageSource) Read(reader io.Reader) ImageSource {
	id := readInt32(reader)
	switch id {
	case 1:
		return ImageSourcePath{
			FfiConverterStringINSTANCE.Read(reader),
		}
	case 2:
		return ImageSourceBytes{
			FfiConverterBytesINSTANCE.Read(reader),
		}
	default:
		panic(fmt.Sprintf("invalid enum value %v in FfiConverterImageSource.Read()", id))
	}
}

func (FfiConverterImageSource) Write(writer io.Writer, value ImageSource) {
	switch variant_value := value.(type) {
	case ImageSourcePath:
		writeInt32(writer, 1)
		FfiConverterStringINSTANCE.Write(writer, variant_value.Path)
	case ImageSourceBytes:
		writeInt32(writer, 2)
		FfiConverterBytesINSTANCE.Write(writer, variant_value.Data)
	default:
		_ = variant_value
		panic(fmt.Sprintf("invalid enum value `%v` in FfiConverterImageSource.Write", value))
	}
}

type FfiDestroyerImageSource struct{}

func (_ FfiDestroyerImageSource) Destroy(value ImageSource) {
	value.Destroy()
}

type OutputGranularity uint

const (
	OutputGranularityLines   OutputGranularity = 1
	OutputGranularityWords   OutputGranularity = 2
	OutputGranularitySpatial OutputGranularity = 3
)

type FfiConverterOutputGranularity struct{}

var FfiConverterOutputGranularityINSTANCE = FfiConverterOutputGranularity{}

func (c FfiConverterOutputGranularity) Lift(rb RustBufferI) OutputGranularity {
	return LiftFromRustBuffer[OutputGranularity](c, rb)
}

func (c FfiConverterOutputGranularity) Lower(value OutputGranularity) C.RustBuffer {
	return LowerIntoRustBuffer[OutputGranularity](c, value)
}

func (c FfiConverterOutputGranularity) LowerExternal(value OutputGranularity) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[OutputGranularity](c, value))
}
func (FfiConverterOutputGranularity) Read(reader io.Reader) OutputGranularity {
	id := readInt32(reader)
	return OutputGranularity(id)
}

func (FfiConverterOutputGranularity) Write(writer io.Writer, value OutputGranularity) {
	writeInt32(writer, int32(value))
}

type FfiDestroyerOutputGranularity struct{}

func (_ FfiDestroyerOutputGranularity) Destroy(value OutputGranularity) {
}

type RustoBindingError struct {
	err error
}

// Convenience method to turn *RustoBindingError into error
// Avoiding treating nil pointer as non nil error interface
func (err *RustoBindingError) AsError() error {
	if err == nil {
		return nil
	} else {
		return err
	}
}

func (err RustoBindingError) Error() string {
	return fmt.Sprintf("RustoBindingError: %s", err.err.Error())
}

func (err RustoBindingError) Unwrap() error {
	return err.err
}

// Err* are used for checking error type with `errors.Is`
var ErrRustoBindingErrorInitialization = fmt.Errorf("RustoBindingErrorInitialization")
var ErrRustoBindingErrorPreprocess = fmt.Errorf("RustoBindingErrorPreprocess")
var ErrRustoBindingErrorInference = fmt.Errorf("RustoBindingErrorInference")
var ErrRustoBindingErrorPostprocess = fmt.Errorf("RustoBindingErrorPostprocess")
var ErrRustoBindingErrorInvalidInput = fmt.Errorf("RustoBindingErrorInvalidInput")

// Variant structs
type RustoBindingErrorInitialization struct {
	Msg string
}

func NewRustoBindingErrorInitialization(
	msg string,
) *RustoBindingError {
	return &RustoBindingError{err: &RustoBindingErrorInitialization{
		Msg: msg}}
}

func (e RustoBindingErrorInitialization) destroy() {
	FfiDestroyerString{}.Destroy(e.Msg)
}

func (err RustoBindingErrorInitialization) Error() string {
	return fmt.Sprint("Initialization",
		": ",

		"Msg=",
		err.Msg,
	)
}

func (self RustoBindingErrorInitialization) Is(target error) bool {
	return target == ErrRustoBindingErrorInitialization
}

type RustoBindingErrorPreprocess struct {
	Msg string
}

func NewRustoBindingErrorPreprocess(
	msg string,
) *RustoBindingError {
	return &RustoBindingError{err: &RustoBindingErrorPreprocess{
		Msg: msg}}
}

func (e RustoBindingErrorPreprocess) destroy() {
	FfiDestroyerString{}.Destroy(e.Msg)
}

func (err RustoBindingErrorPreprocess) Error() string {
	return fmt.Sprint("Preprocess",
		": ",

		"Msg=",
		err.Msg,
	)
}

func (self RustoBindingErrorPreprocess) Is(target error) bool {
	return target == ErrRustoBindingErrorPreprocess
}

type RustoBindingErrorInference struct {
	Msg string
}

func NewRustoBindingErrorInference(
	msg string,
) *RustoBindingError {
	return &RustoBindingError{err: &RustoBindingErrorInference{
		Msg: msg}}
}

func (e RustoBindingErrorInference) destroy() {
	FfiDestroyerString{}.Destroy(e.Msg)
}

func (err RustoBindingErrorInference) Error() string {
	return fmt.Sprint("Inference",
		": ",

		"Msg=",
		err.Msg,
	)
}

func (self RustoBindingErrorInference) Is(target error) bool {
	return target == ErrRustoBindingErrorInference
}

type RustoBindingErrorPostprocess struct {
	Msg string
}

func NewRustoBindingErrorPostprocess(
	msg string,
) *RustoBindingError {
	return &RustoBindingError{err: &RustoBindingErrorPostprocess{
		Msg: msg}}
}

func (e RustoBindingErrorPostprocess) destroy() {
	FfiDestroyerString{}.Destroy(e.Msg)
}

func (err RustoBindingErrorPostprocess) Error() string {
	return fmt.Sprint("Postprocess",
		": ",

		"Msg=",
		err.Msg,
	)
}

func (self RustoBindingErrorPostprocess) Is(target error) bool {
	return target == ErrRustoBindingErrorPostprocess
}

type RustoBindingErrorInvalidInput struct {
	Msg string
}

func NewRustoBindingErrorInvalidInput(
	msg string,
) *RustoBindingError {
	return &RustoBindingError{err: &RustoBindingErrorInvalidInput{
		Msg: msg}}
}

func (e RustoBindingErrorInvalidInput) destroy() {
	FfiDestroyerString{}.Destroy(e.Msg)
}

func (err RustoBindingErrorInvalidInput) Error() string {
	return fmt.Sprint("InvalidInput",
		": ",

		"Msg=",
		err.Msg,
	)
}

func (self RustoBindingErrorInvalidInput) Is(target error) bool {
	return target == ErrRustoBindingErrorInvalidInput
}

type FfiConverterRustoBindingError struct{}

var FfiConverterRustoBindingErrorINSTANCE = FfiConverterRustoBindingError{}

func (c FfiConverterRustoBindingError) Lift(eb RustBufferI) *RustoBindingError {
	return LiftFromRustBuffer[*RustoBindingError](c, eb)
}

func (c FfiConverterRustoBindingError) Lower(value *RustoBindingError) C.RustBuffer {
	return LowerIntoRustBuffer[*RustoBindingError](c, value)
}

func (c FfiConverterRustoBindingError) LowerExternal(value *RustoBindingError) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*RustoBindingError](c, value))
}

func (c FfiConverterRustoBindingError) Read(reader io.Reader) *RustoBindingError {
	errorID := readUint32(reader)

	switch errorID {
	case 1:
		return &RustoBindingError{&RustoBindingErrorInitialization{
			Msg: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 2:
		return &RustoBindingError{&RustoBindingErrorPreprocess{
			Msg: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 3:
		return &RustoBindingError{&RustoBindingErrorInference{
			Msg: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 4:
		return &RustoBindingError{&RustoBindingErrorPostprocess{
			Msg: FfiConverterStringINSTANCE.Read(reader),
		}}
	case 5:
		return &RustoBindingError{&RustoBindingErrorInvalidInput{
			Msg: FfiConverterStringINSTANCE.Read(reader),
		}}
	default:
		panic(fmt.Sprintf("Unknown error code %d in FfiConverterRustoBindingError.Read()", errorID))
	}
}

func (c FfiConverterRustoBindingError) Write(writer io.Writer, value *RustoBindingError) {
	switch variantValue := value.err.(type) {
	case *RustoBindingErrorInitialization:
		writeInt32(writer, 1)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Msg)
	case *RustoBindingErrorPreprocess:
		writeInt32(writer, 2)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Msg)
	case *RustoBindingErrorInference:
		writeInt32(writer, 3)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Msg)
	case *RustoBindingErrorPostprocess:
		writeInt32(writer, 4)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Msg)
	case *RustoBindingErrorInvalidInput:
		writeInt32(writer, 5)
		FfiConverterStringINSTANCE.Write(writer, variantValue.Msg)
	default:
		_ = variantValue
		panic(fmt.Sprintf("invalid error value `%v` in FfiConverterRustoBindingError.Write", value))
	}
}

type FfiDestroyerRustoBindingError struct{}

func (_ FfiDestroyerRustoBindingError) Destroy(value *RustoBindingError) {
	switch variantValue := value.err.(type) {
	case RustoBindingErrorInitialization:
		variantValue.destroy()
	case RustoBindingErrorPreprocess:
		variantValue.destroy()
	case RustoBindingErrorInference:
		variantValue.destroy()
	case RustoBindingErrorPostprocess:
		variantValue.destroy()
	case RustoBindingErrorInvalidInput:
		variantValue.destroy()
	default:
		_ = variantValue
		panic(fmt.Sprintf("invalid error value `%v` in FfiDestroyerRustoBindingError.Destroy", value))
	}
}

type FfiConverterOptionalInt32 struct{}

var FfiConverterOptionalInt32INSTANCE = FfiConverterOptionalInt32{}

func (c FfiConverterOptionalInt32) Lift(rb RustBufferI) *int32 {
	return LiftFromRustBuffer[*int32](c, rb)
}

func (_ FfiConverterOptionalInt32) Read(reader io.Reader) *int32 {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterInt32INSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalInt32) Lower(value *int32) C.RustBuffer {
	return LowerIntoRustBuffer[*int32](c, value)
}

func (c FfiConverterOptionalInt32) LowerExternal(value *int32) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*int32](c, value))
}

func (_ FfiConverterOptionalInt32) Write(writer io.Writer, value *int32) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterInt32INSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalInt32 struct{}

func (_ FfiDestroyerOptionalInt32) Destroy(value *int32) {
	if value != nil {
		FfiDestroyerInt32{}.Destroy(*value)
	}
}

type FfiConverterOptionalFloat32 struct{}

var FfiConverterOptionalFloat32INSTANCE = FfiConverterOptionalFloat32{}

func (c FfiConverterOptionalFloat32) Lift(rb RustBufferI) *float32 {
	return LiftFromRustBuffer[*float32](c, rb)
}

func (_ FfiConverterOptionalFloat32) Read(reader io.Reader) *float32 {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterFloat32INSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalFloat32) Lower(value *float32) C.RustBuffer {
	return LowerIntoRustBuffer[*float32](c, value)
}

func (c FfiConverterOptionalFloat32) LowerExternal(value *float32) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*float32](c, value))
}

func (_ FfiConverterOptionalFloat32) Write(writer io.Writer, value *float32) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterFloat32INSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalFloat32 struct{}

func (_ FfiDestroyerOptionalFloat32) Destroy(value *float32) {
	if value != nil {
		FfiDestroyerFloat32{}.Destroy(*value)
	}
}

type FfiConverterOptionalBool struct{}

var FfiConverterOptionalBoolINSTANCE = FfiConverterOptionalBool{}

func (c FfiConverterOptionalBool) Lift(rb RustBufferI) *bool {
	return LiftFromRustBuffer[*bool](c, rb)
}

func (_ FfiConverterOptionalBool) Read(reader io.Reader) *bool {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterBoolINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalBool) Lower(value *bool) C.RustBuffer {
	return LowerIntoRustBuffer[*bool](c, value)
}

func (c FfiConverterOptionalBool) LowerExternal(value *bool) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*bool](c, value))
}

func (_ FfiConverterOptionalBool) Write(writer io.Writer, value *bool) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterBoolINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalBool struct{}

func (_ FfiDestroyerOptionalBool) Destroy(value *bool) {
	if value != nil {
		FfiDestroyerBool{}.Destroy(*value)
	}
}

type FfiConverterOptionalString struct{}

var FfiConverterOptionalStringINSTANCE = FfiConverterOptionalString{}

func (c FfiConverterOptionalString) Lift(rb RustBufferI) *string {
	return LiftFromRustBuffer[*string](c, rb)
}

func (_ FfiConverterOptionalString) Read(reader io.Reader) *string {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterStringINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalString) Lower(value *string) C.RustBuffer {
	return LowerIntoRustBuffer[*string](c, value)
}

func (c FfiConverterOptionalString) LowerExternal(value *string) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*string](c, value))
}

func (_ FfiConverterOptionalString) Write(writer io.Writer, value *string) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterStringINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalString struct{}

func (_ FfiDestroyerOptionalString) Destroy(value *string) {
	if value != nil {
		FfiDestroyerString{}.Destroy(*value)
	}
}

type FfiConverterOptionalCalibrationOptions struct{}

var FfiConverterOptionalCalibrationOptionsINSTANCE = FfiConverterOptionalCalibrationOptions{}

func (c FfiConverterOptionalCalibrationOptions) Lift(rb RustBufferI) *CalibrationOptions {
	return LiftFromRustBuffer[*CalibrationOptions](c, rb)
}

func (_ FfiConverterOptionalCalibrationOptions) Read(reader io.Reader) *CalibrationOptions {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterCalibrationOptionsINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalCalibrationOptions) Lower(value *CalibrationOptions) C.RustBuffer {
	return LowerIntoRustBuffer[*CalibrationOptions](c, value)
}

func (c FfiConverterOptionalCalibrationOptions) LowerExternal(value *CalibrationOptions) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*CalibrationOptions](c, value))
}

func (_ FfiConverterOptionalCalibrationOptions) Write(writer io.Writer, value *CalibrationOptions) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterCalibrationOptionsINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalCalibrationOptions struct{}

func (_ FfiDestroyerOptionalCalibrationOptions) Destroy(value *CalibrationOptions) {
	if value != nil {
		FfiDestroyerCalibrationOptions{}.Destroy(*value)
	}
}

type FfiConverterOptionalClassificationConfig struct{}

var FfiConverterOptionalClassificationConfigINSTANCE = FfiConverterOptionalClassificationConfig{}

func (c FfiConverterOptionalClassificationConfig) Lift(rb RustBufferI) *ClassificationConfig {
	return LiftFromRustBuffer[*ClassificationConfig](c, rb)
}

func (_ FfiConverterOptionalClassificationConfig) Read(reader io.Reader) *ClassificationConfig {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterClassificationConfigINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalClassificationConfig) Lower(value *ClassificationConfig) C.RustBuffer {
	return LowerIntoRustBuffer[*ClassificationConfig](c, value)
}

func (c FfiConverterOptionalClassificationConfig) LowerExternal(value *ClassificationConfig) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*ClassificationConfig](c, value))
}

func (_ FfiConverterOptionalClassificationConfig) Write(writer io.Writer, value *ClassificationConfig) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterClassificationConfigINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalClassificationConfig struct{}

func (_ FfiDestroyerOptionalClassificationConfig) Destroy(value *ClassificationConfig) {
	if value != nil {
		FfiDestroyerClassificationConfig{}.Destroy(*value)
	}
}

type FfiConverterOptionalDetectionConfig struct{}

var FfiConverterOptionalDetectionConfigINSTANCE = FfiConverterOptionalDetectionConfig{}

func (c FfiConverterOptionalDetectionConfig) Lift(rb RustBufferI) *DetectionConfig {
	return LiftFromRustBuffer[*DetectionConfig](c, rb)
}

func (_ FfiConverterOptionalDetectionConfig) Read(reader io.Reader) *DetectionConfig {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterDetectionConfigINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalDetectionConfig) Lower(value *DetectionConfig) C.RustBuffer {
	return LowerIntoRustBuffer[*DetectionConfig](c, value)
}

func (c FfiConverterOptionalDetectionConfig) LowerExternal(value *DetectionConfig) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*DetectionConfig](c, value))
}

func (_ FfiConverterOptionalDetectionConfig) Write(writer io.Writer, value *DetectionConfig) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterDetectionConfigINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalDetectionConfig struct{}

func (_ FfiDestroyerOptionalDetectionConfig) Destroy(value *DetectionConfig) {
	if value != nil {
		FfiDestroyerDetectionConfig{}.Destroy(*value)
	}
}

type FfiConverterOptionalDetectionRunOptions struct{}

var FfiConverterOptionalDetectionRunOptionsINSTANCE = FfiConverterOptionalDetectionRunOptions{}

func (c FfiConverterOptionalDetectionRunOptions) Lift(rb RustBufferI) *DetectionRunOptions {
	return LiftFromRustBuffer[*DetectionRunOptions](c, rb)
}

func (_ FfiConverterOptionalDetectionRunOptions) Read(reader io.Reader) *DetectionRunOptions {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterDetectionRunOptionsINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalDetectionRunOptions) Lower(value *DetectionRunOptions) C.RustBuffer {
	return LowerIntoRustBuffer[*DetectionRunOptions](c, value)
}

func (c FfiConverterOptionalDetectionRunOptions) LowerExternal(value *DetectionRunOptions) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*DetectionRunOptions](c, value))
}

func (_ FfiConverterOptionalDetectionRunOptions) Write(writer io.Writer, value *DetectionRunOptions) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterDetectionRunOptionsINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalDetectionRunOptions struct{}

func (_ FfiDestroyerOptionalDetectionRunOptions) Destroy(value *DetectionRunOptions) {
	if value != nil {
		FfiDestroyerDetectionRunOptions{}.Destroy(*value)
	}
}

type FfiConverterOptionalOptimizationOptions struct{}

var FfiConverterOptionalOptimizationOptionsINSTANCE = FfiConverterOptionalOptimizationOptions{}

func (c FfiConverterOptionalOptimizationOptions) Lift(rb RustBufferI) *OptimizationOptions {
	return LiftFromRustBuffer[*OptimizationOptions](c, rb)
}

func (_ FfiConverterOptionalOptimizationOptions) Read(reader io.Reader) *OptimizationOptions {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterOptimizationOptionsINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalOptimizationOptions) Lower(value *OptimizationOptions) C.RustBuffer {
	return LowerIntoRustBuffer[*OptimizationOptions](c, value)
}

func (c FfiConverterOptionalOptimizationOptions) LowerExternal(value *OptimizationOptions) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*OptimizationOptions](c, value))
}

func (_ FfiConverterOptionalOptimizationOptions) Write(writer io.Writer, value *OptimizationOptions) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterOptimizationOptionsINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalOptimizationOptions struct{}

func (_ FfiDestroyerOptionalOptimizationOptions) Destroy(value *OptimizationOptions) {
	if value != nil {
		FfiDestroyerOptimizationOptions{}.Destroy(*value)
	}
}

type FfiConverterOptionalOrientationConfig struct{}

var FfiConverterOptionalOrientationConfigINSTANCE = FfiConverterOptionalOrientationConfig{}

func (c FfiConverterOptionalOrientationConfig) Lift(rb RustBufferI) *OrientationConfig {
	return LiftFromRustBuffer[*OrientationConfig](c, rb)
}

func (_ FfiConverterOptionalOrientationConfig) Read(reader io.Reader) *OrientationConfig {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterOrientationConfigINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalOrientationConfig) Lower(value *OrientationConfig) C.RustBuffer {
	return LowerIntoRustBuffer[*OrientationConfig](c, value)
}

func (c FfiConverterOptionalOrientationConfig) LowerExternal(value *OrientationConfig) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*OrientationConfig](c, value))
}

func (_ FfiConverterOptionalOrientationConfig) Write(writer io.Writer, value *OrientationConfig) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterOrientationConfigINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalOrientationConfig struct{}

func (_ FfiDestroyerOptionalOrientationConfig) Destroy(value *OrientationConfig) {
	if value != nil {
		FfiDestroyerOrientationConfig{}.Destroy(*value)
	}
}

type FfiConverterOptionalPostprocessRunOptions struct{}

var FfiConverterOptionalPostprocessRunOptionsINSTANCE = FfiConverterOptionalPostprocessRunOptions{}

func (c FfiConverterOptionalPostprocessRunOptions) Lift(rb RustBufferI) *PostprocessRunOptions {
	return LiftFromRustBuffer[*PostprocessRunOptions](c, rb)
}

func (_ FfiConverterOptionalPostprocessRunOptions) Read(reader io.Reader) *PostprocessRunOptions {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterPostprocessRunOptionsINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalPostprocessRunOptions) Lower(value *PostprocessRunOptions) C.RustBuffer {
	return LowerIntoRustBuffer[*PostprocessRunOptions](c, value)
}

func (c FfiConverterOptionalPostprocessRunOptions) LowerExternal(value *PostprocessRunOptions) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*PostprocessRunOptions](c, value))
}

func (_ FfiConverterOptionalPostprocessRunOptions) Write(writer io.Writer, value *PostprocessRunOptions) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterPostprocessRunOptionsINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalPostprocessRunOptions struct{}

func (_ FfiDestroyerOptionalPostprocessRunOptions) Destroy(value *PostprocessRunOptions) {
	if value != nil {
		FfiDestroyerPostprocessRunOptions{}.Destroy(*value)
	}
}

type FfiConverterOptionalRecognitionConfig struct{}

var FfiConverterOptionalRecognitionConfigINSTANCE = FfiConverterOptionalRecognitionConfig{}

func (c FfiConverterOptionalRecognitionConfig) Lift(rb RustBufferI) *RecognitionConfig {
	return LiftFromRustBuffer[*RecognitionConfig](c, rb)
}

func (_ FfiConverterOptionalRecognitionConfig) Read(reader io.Reader) *RecognitionConfig {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterRecognitionConfigINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalRecognitionConfig) Lower(value *RecognitionConfig) C.RustBuffer {
	return LowerIntoRustBuffer[*RecognitionConfig](c, value)
}

func (c FfiConverterOptionalRecognitionConfig) LowerExternal(value *RecognitionConfig) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*RecognitionConfig](c, value))
}

func (_ FfiConverterOptionalRecognitionConfig) Write(writer io.Writer, value *RecognitionConfig) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterRecognitionConfigINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalRecognitionConfig struct{}

func (_ FfiDestroyerOptionalRecognitionConfig) Destroy(value *RecognitionConfig) {
	if value != nil {
		FfiDestroyerRecognitionConfig{}.Destroy(*value)
	}
}

type FfiConverterOptionalUnwarpConfig struct{}

var FfiConverterOptionalUnwarpConfigINSTANCE = FfiConverterOptionalUnwarpConfig{}

func (c FfiConverterOptionalUnwarpConfig) Lift(rb RustBufferI) *UnwarpConfig {
	return LiftFromRustBuffer[*UnwarpConfig](c, rb)
}

func (_ FfiConverterOptionalUnwarpConfig) Read(reader io.Reader) *UnwarpConfig {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterUnwarpConfigINSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalUnwarpConfig) Lower(value *UnwarpConfig) C.RustBuffer {
	return LowerIntoRustBuffer[*UnwarpConfig](c, value)
}

func (c FfiConverterOptionalUnwarpConfig) LowerExternal(value *UnwarpConfig) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*UnwarpConfig](c, value))
}

func (_ FfiConverterOptionalUnwarpConfig) Write(writer io.Writer, value *UnwarpConfig) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterUnwarpConfigINSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalUnwarpConfig struct{}

func (_ FfiDestroyerOptionalUnwarpConfig) Destroy(value *UnwarpConfig) {
	if value != nil {
		FfiDestroyerUnwarpConfig{}.Destroy(*value)
	}
}

type FfiConverterOptionalSequenceFloat32 struct{}

var FfiConverterOptionalSequenceFloat32INSTANCE = FfiConverterOptionalSequenceFloat32{}

func (c FfiConverterOptionalSequenceFloat32) Lift(rb RustBufferI) *[]float32 {
	return LiftFromRustBuffer[*[]float32](c, rb)
}

func (_ FfiConverterOptionalSequenceFloat32) Read(reader io.Reader) *[]float32 {
	if readInt8(reader) == 0 {
		return nil
	}
	temp := FfiConverterSequenceFloat32INSTANCE.Read(reader)
	return &temp
}

func (c FfiConverterOptionalSequenceFloat32) Lower(value *[]float32) C.RustBuffer {
	return LowerIntoRustBuffer[*[]float32](c, value)
}

func (c FfiConverterOptionalSequenceFloat32) LowerExternal(value *[]float32) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[*[]float32](c, value))
}

func (_ FfiConverterOptionalSequenceFloat32) Write(writer io.Writer, value *[]float32) {
	if value == nil {
		writeInt8(writer, 0)
	} else {
		writeInt8(writer, 1)
		FfiConverterSequenceFloat32INSTANCE.Write(writer, *value)
	}
}

type FfiDestroyerOptionalSequenceFloat32 struct{}

func (_ FfiDestroyerOptionalSequenceFloat32) Destroy(value *[]float32) {
	if value != nil {
		FfiDestroyerSequenceFloat32{}.Destroy(*value)
	}
}

type FfiConverterSequenceFloat32 struct{}

var FfiConverterSequenceFloat32INSTANCE = FfiConverterSequenceFloat32{}

func (c FfiConverterSequenceFloat32) Lift(rb RustBufferI) []float32 {
	return LiftFromRustBuffer[[]float32](c, rb)
}

func (c FfiConverterSequenceFloat32) Read(reader io.Reader) []float32 {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]float32, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterFloat32INSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceFloat32) Lower(value []float32) C.RustBuffer {
	return LowerIntoRustBuffer[[]float32](c, value)
}

func (c FfiConverterSequenceFloat32) LowerExternal(value []float32) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]float32](c, value))
}

func (c FfiConverterSequenceFloat32) Write(writer io.Writer, value []float32) {
	if len(value) > math.MaxInt32 {
		panic("[]float32 is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterFloat32INSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceFloat32 struct{}

func (FfiDestroyerSequenceFloat32) Destroy(sequence []float32) {
	for _, value := range sequence {
		FfiDestroyerFloat32{}.Destroy(value)
	}
}

type FfiConverterSequencePoint2D struct{}

var FfiConverterSequencePoint2DINSTANCE = FfiConverterSequencePoint2D{}

func (c FfiConverterSequencePoint2D) Lift(rb RustBufferI) []Point2D {
	return LiftFromRustBuffer[[]Point2D](c, rb)
}

func (c FfiConverterSequencePoint2D) Read(reader io.Reader) []Point2D {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]Point2D, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterPoint2DINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequencePoint2D) Lower(value []Point2D) C.RustBuffer {
	return LowerIntoRustBuffer[[]Point2D](c, value)
}

func (c FfiConverterSequencePoint2D) LowerExternal(value []Point2D) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]Point2D](c, value))
}

func (c FfiConverterSequencePoint2D) Write(writer io.Writer, value []Point2D) {
	if len(value) > math.MaxInt32 {
		panic("[]Point2D is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterPoint2DINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequencePoint2D struct{}

func (FfiDestroyerSequencePoint2D) Destroy(sequence []Point2D) {
	for _, value := range sequence {
		FfiDestroyerPoint2D{}.Destroy(value)
	}
}

type FfiConverterSequenceTextResult struct{}

var FfiConverterSequenceTextResultINSTANCE = FfiConverterSequenceTextResult{}

func (c FfiConverterSequenceTextResult) Lift(rb RustBufferI) []TextResult {
	return LiftFromRustBuffer[[]TextResult](c, rb)
}

func (c FfiConverterSequenceTextResult) Read(reader io.Reader) []TextResult {
	length := readInt32(reader)
	if length == 0 {
		return nil
	}
	result := make([]TextResult, 0, length)
	for i := int32(0); i < length; i++ {
		result = append(result, FfiConverterTextResultINSTANCE.Read(reader))
	}
	return result
}

func (c FfiConverterSequenceTextResult) Lower(value []TextResult) C.RustBuffer {
	return LowerIntoRustBuffer[[]TextResult](c, value)
}

func (c FfiConverterSequenceTextResult) LowerExternal(value []TextResult) ExternalCRustBuffer {
	return RustBufferFromC(LowerIntoRustBuffer[[]TextResult](c, value))
}

func (c FfiConverterSequenceTextResult) Write(writer io.Writer, value []TextResult) {
	if len(value) > math.MaxInt32 {
		panic("[]TextResult is too large to fit into Int32")
	}

	writeInt32(writer, int32(len(value)))
	for _, item := range value {
		FfiConverterTextResultINSTANCE.Write(writer, item)
	}
}

type FfiDestroyerSequenceTextResult struct{}

func (FfiDestroyerSequenceTextResult) Destroy(sequence []TextResult) {
	for _, value := range sequence {
		FfiDestroyerTextResult{}.Destroy(value)
	}
}

func CreateInitializeConfigPpv6(detModelPath string, recModelPath string, dictPath string) InitializeConfig {
	return FfiConverterInitializeConfigINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_rusto_fn_func_create_initialize_config_ppv6(FfiConverterStringINSTANCE.Lower(detModelPath), FfiConverterStringINSTANCE.Lower(recModelPath), FfiConverterStringINSTANCE.Lower(dictPath), _uniffiStatus),
		}
	}))
}

func DefaultOcrRunOptions() OcrRunOptions {
	return FfiConverterOcrRunOptionsINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_rusto_fn_func_default_ocr_run_options(_uniffiStatus),
		}
	}))
}

func RustoVersion() string {
	return FfiConverterStringINSTANCE.Lift(rustCall(func(_uniffiStatus *C.RustCallStatus) RustBufferI {
		return GoRustBuffer{
			inner: C.uniffi_rusto_fn_func_rusto_version(_uniffiStatus),
		}
	}))
}
