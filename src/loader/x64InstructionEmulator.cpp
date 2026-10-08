#include "loader/x64InstructionEmulator.h"

#include "common/common.h"

#include <Zydis/Zydis.h>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#if !defined(__APPLE__)
#include <emmintrin.h>
#include <xmmintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#elif defined(__APPLE__)
#include <sys/ucontext.h>
#else
#include <sched.h>
#include <ucontext.h>
#endif

namespace Loader::X64InstructionEmulator {

static uint64_t ExtractBitField(uint64_t value, uint32_t length, uint32_t index) {
	length &= 0x3fu;
	index &= 0x3fu;

	if (length == 0) {
		length = 64;
	}

	if (index >= 64) {
		return 0;
	}

	auto available = 64u - index;
	if (length > available) {
		length = available;
	}

	const uint64_t mask = (length == 64 ? UINT64_MAX : ((uint64_t {1} << length) - 1u));
	return (value >> index) & mask;
}

static uint64_t InsertBitField(uint64_t dst, uint64_t src, uint32_t length, uint32_t index) {
	length &= 0x3fu;
	index &= 0x3fu;

	if (length == 0) {
		length = 64;
	}

	if (index >= 64) {
		return dst;
	}

	auto available = 64u - index;
	if (length > available) {
		length = available;
	}

	const uint64_t mask        = (length == 64 ? UINT64_MAX : ((uint64_t {1} << length) - 1u));
	const uint64_t shifted     = (index == 0 ? mask : (mask << index));
	const uint64_t src_shifted = (src & mask) << index;

	return (dst & ~shifted) | src_shifted;
}

struct XmmWords {
	uint32_t w[4];
};

static void Sha1Msg1(XmmWords& dest, const XmmWords& src2) {
	const uint32_t w0 = dest.w[3];
	const uint32_t w1 = dest.w[2];
	const uint32_t w2 = dest.w[1];
	const uint32_t w3 = dest.w[0];
	const uint32_t w4 = src2.w[3];
	const uint32_t w5 = src2.w[2];
	dest.w[3]         = w2 ^ w0;
	dest.w[2]         = w3 ^ w1;
	dest.w[1]         = w4 ^ w2;
	dest.w[0]         = w5 ^ w3;
}

static void Sha1Msg2(XmmWords& dest, const XmmWords& src2) {
	const uint32_t w13 = src2.w[2];
	const uint32_t w14 = src2.w[1];
	const uint32_t w15 = src2.w[0];
	const uint32_t w16 = std::rotl(dest.w[3] ^ w13, 1);
	const uint32_t w17 = std::rotl(dest.w[2] ^ w14, 1);
	const uint32_t w18 = std::rotl(dest.w[1] ^ w15, 1);
	const uint32_t w19 = std::rotl(dest.w[0] ^ w16, 1);
	dest.w[3]          = w16;
	dest.w[2]          = w17;
	dest.w[1]          = w18;
	dest.w[0]          = w19;
}

static void Sha1Nexte(XmmWords& dest, const XmmWords& src2) {
	const uint32_t tmp = std::rotl(dest.w[3], 30);
	dest.w[3]          = src2.w[3] + tmp;
	dest.w[2]          = src2.w[2];
	dest.w[1]          = src2.w[1];
	dest.w[0]          = src2.w[0];
}

static uint32_t Sha1RoundFunc(uint8_t group, uint32_t b, uint32_t c, uint32_t d) {
	switch (group & 3u) {
		case 0: return (b & c) ^ ((~b) & d);
		case 1: return b ^ c ^ d;
		case 2: return (b & c) ^ (b & d) ^ (c & d);
		default: return b ^ c ^ d;
	}
}

static uint32_t Sha1RoundConstant(uint8_t group) {
	switch (group & 3u) {
		case 0: return 0x5a827999u;
		case 1: return 0x6ed9eba1u;
		case 2: return 0x8f1bbcdcu;
		default: return 0xca62c1d6u;
	}
}

static void Sha1Rnds4(XmmWords& dest, const XmmWords& src2, uint8_t imm8) {
	const uint8_t  group = imm8 & 3u;
	const uint32_t k     = Sha1RoundConstant(group);
	const uint32_t w[4]  = {src2.w[3], src2.w[2], src2.w[1], src2.w[0]};

	uint32_t a = dest.w[3];
	uint32_t b = dest.w[2];
	uint32_t c = dest.w[1];
	uint32_t d = dest.w[0];
	uint32_t e = 0;

	for (unsigned int round = 0; round < 4u; round++) {
		uint32_t term = Sha1RoundFunc(group, b, c, d) + std::rotl(a, 5) + w[round] + k;
		if (round > 0u) {
			term += e;
		}
		const uint32_t a1 = term;
		e                 = d;
		d                 = c;
		c                 = std::rotl(b, 30);
		b                 = a;
		a                 = a1;
	}

	dest.w[3] = a;
	dest.w[2] = b;
	dest.w[1] = c;
	dest.w[0] = d;
}

static uint32_t Sha256Sigma0(uint32_t x) {
	return std::rotr(x, 7) ^ std::rotr(x, 18) ^ (x >> 3u);
}

static uint32_t Sha256Sigma1(uint32_t x) {
	return std::rotr(x, 17) ^ std::rotr(x, 19) ^ (x >> 10u);
}

static uint32_t Sha256Sum0(uint32_t x) {
	return std::rotr(x, 2) ^ std::rotr(x, 13) ^ std::rotr(x, 22);
}

static uint32_t Sha256Sum1(uint32_t x) {
	return std::rotr(x, 6) ^ std::rotr(x, 11) ^ std::rotr(x, 25);
}

static uint32_t Sha256Ch(uint32_t e, uint32_t f, uint32_t g) {
	return (e & f) ^ ((~e) & g);
}

static uint32_t Sha256Maj(uint32_t a, uint32_t b, uint32_t c) {
	return (a & b) ^ (a & c) ^ (b & c);
}

static void Sha256Msg1(XmmWords& dest, const XmmWords& src2) {
	const uint32_t w4 = src2.w[0];
	const uint32_t w3 = dest.w[3];
	const uint32_t w2 = dest.w[2];
	const uint32_t w1 = dest.w[1];
	const uint32_t w0 = dest.w[0];
	dest.w[3]         = w3 + Sha256Sigma0(w4);
	dest.w[2]         = w2 + Sha256Sigma0(w3);
	dest.w[1]         = w1 + Sha256Sigma0(w2);
	dest.w[0]         = w0 + Sha256Sigma0(w1);
}

static void Sha256Msg2(XmmWords& dest, const XmmWords& src2) {
	const uint32_t w14 = src2.w[2];
	const uint32_t w15 = src2.w[3];
	const uint32_t w16 = dest.w[0] + Sha256Sigma1(w14);
	const uint32_t w17 = dest.w[1] + Sha256Sigma1(w15);
	const uint32_t w18 = dest.w[2] + Sha256Sigma1(w16);
	const uint32_t w19 = dest.w[3] + Sha256Sigma1(w17);
	dest.w[3]          = w19;
	dest.w[2]          = w18;
	dest.w[1]          = w17;
	dest.w[0]          = w16;
}

static void Sha256Rnds2(XmmWords& dest, const XmmWords& src2, const XmmWords& xmm0) {
	uint32_t a = src2.w[3];
	uint32_t b = src2.w[2];
	uint32_t c = dest.w[3];
	uint32_t d = dest.w[2];
	uint32_t e = src2.w[1];
	uint32_t f = src2.w[0];
	uint32_t g = dest.w[1];
	uint32_t h = dest.w[0];

	for (unsigned int round = 0; round < 2u; round++) {
		const uint32_t wk = xmm0.w[round];
		const uint32_t t1 = Sha256Ch(e, f, g) + Sha256Sum1(e) + wk + h;
		const uint32_t t2 = Sha256Maj(a, b, c) + Sha256Sum0(a);
		const uint32_t a1 = t1 + t2;
		const uint32_t e1 = t1 + d;
		const uint32_t b1 = a;
		const uint32_t c1 = b;
		const uint32_t d1 = c;
		const uint32_t f1 = e;
		const uint32_t g1 = f;
		const uint32_t h1 = g;
		a                 = a1;
		b                 = b1;
		c                 = c1;
		d                 = d1;
		e                 = e1;
		f                 = f1;
		g                 = g1;
		h                 = h1;
	}

	dest.w[3] = a;
	dest.w[2] = b;
	dest.w[1] = e;
	dest.w[0] = f;
}

struct ShaNiInsn {
	uint8_t escape;
	uint8_t opcode;
	uint8_t imm8;
	uint8_t rex;
	size_t  modrm_offset;
	size_t  length;
};

static bool DecodeShaNiInsn(const uint8_t* rip, ShaNiInsn& insn) {
	size_t  offset = 0;
	uint8_t rex    = 0;
	if ((rip[0] & 0xf0u) == 0x40u) {
		rex    = rip[0];
		offset = 1;
	}

	if (rip[offset] != 0x0f) {
		return false;
	}

	if (rip[offset + 1] == 0x38) {
		const uint8_t op = rip[offset + 2];
		if (op != 0xc8 && op != 0xc9 && op != 0xca && op != 0xcb && op != 0xcc && op != 0xcd) {
			return false;
		}
		insn.escape       = 0x38;
		insn.opcode       = op;
		insn.imm8         = 0;
		insn.rex          = rex;
		insn.modrm_offset = offset + 3;
	} else if (rip[offset + 1] == 0x3a && rip[offset + 2] == 0xcc) {
		insn.escape       = 0x3a;
		insn.opcode       = 0xcc;
		insn.rex          = rex;
		insn.modrm_offset = offset + 3;
	} else {
		return false;
	}

	const uint8_t modrm = rip[insn.modrm_offset];
	const uint8_t mod   = modrm >> 6u;
	const uint8_t rm    = modrm & 0x07u;
	size_t        end   = insn.modrm_offset + 1;

	if (mod != 3u) {
		uint8_t sib_base = 0xffu;
		if (rm == 4u) {
			sib_base = rip[end] & 0x07u;
			end++;
		}

		if (mod == 0u && (rm == 5u || (rm == 4u && sib_base == 5u))) {
			end += 4;
		} else if (mod == 1u) {
			end++;
		} else if (mod == 2u) {
			end += 4;
		}
	}

	if (insn.escape == 0x3a) {
		insn.imm8 = rip[end];
		end++;
	}

	insn.length = end;
	return true;
}

static bool ShaNiModrmIsRegister(uint8_t modrm) {
	return (modrm & 0xc0u) == 0xc0u;
}

static uint8_t ShaNiRegIndex(uint8_t modrm, uint8_t rex, bool reg_field) {
	if (reg_field) {
		return ((modrm >> 3u) & 0x07u) | ((rex & 0x04u) << 1u);
	}
	return (modrm & 0x07u) | ((rex & 0x01u) << 3u);
}

static bool ResolveShaNiMemoryAddress(const uint8_t* rip, const ShaNiInsn&    insn,
                                      const uint64_t (&gpr)[16], const void*& address) {
	const uint8_t modrm = rip[insn.modrm_offset];
	const uint8_t mod   = modrm >> 6u;
	const uint8_t rm    = modrm & 0x07u;
	if (mod == 3u) {
		return false;
	}

	size_t   offset = insn.modrm_offset + 1;
	uint64_t result = 0;

	if (rm == 4u) {
		const uint8_t sib       = rip[offset++];
		const uint8_t scale     = sib >> 6u;
		const uint8_t index_low = (sib >> 3u) & 0x07u;
		const uint8_t base_low  = sib & 0x07u;
		const bool    has_index = index_low != 4u || (insn.rex & 0x02u) != 0;
		const bool    has_base  = mod != 0u || base_low != 5u;

		if (has_base) {
			const uint8_t base = base_low | ((insn.rex & 0x01u) << 3u);
			result += gpr[base];
		}
		if (has_index) {
			const uint8_t index = index_low | ((insn.rex & 0x02u) << 2u);
			result += gpr[index] << scale;
		}

		if (!has_base) {
			int32_t displacement = 0;
			std::memcpy(&displacement, rip + offset, sizeof(displacement));
			result += static_cast<uint64_t>(static_cast<int64_t>(displacement));
			offset += sizeof(displacement);
		}
	} else if (mod == 0u && rm == 5u) {
		int32_t displacement = 0;
		std::memcpy(&displacement, rip + offset, sizeof(displacement));
		result = reinterpret_cast<uint64_t>(rip + insn.length) +
		         static_cast<uint64_t>(static_cast<int64_t>(displacement));
		offset += sizeof(displacement);
	} else {
		const uint8_t base = rm | ((insn.rex & 0x01u) << 3u);
		result             = gpr[base];
	}

	if (mod == 1u) {
		const auto displacement = static_cast<int8_t>(rip[offset]);
		result += static_cast<uint64_t>(static_cast<int64_t>(displacement));
	} else if (mod == 2u) {
		int32_t displacement = 0;
		std::memcpy(&displacement, rip + offset, sizeof(displacement));
		result += static_cast<uint64_t>(static_cast<int64_t>(displacement));
	}

	address = reinterpret_cast<const void*>(result);
	return true;
}

static bool ExecuteShaNiInsn(const ShaNiInsn& insn, const XmmWords& src2, const XmmWords& xmm0,
                             XmmWords& dest) {
	if (insn.escape == 0x3a && insn.opcode == 0xcc) {
		Sha1Rnds4(dest, src2, insn.imm8);
		return true;
	}

	switch (insn.opcode) {
		case 0xc8: Sha1Nexte(dest, src2); return true;
		case 0xc9: Sha1Msg1(dest, src2); return true;
		case 0xca: Sha1Msg2(dest, src2); return true;
		case 0xcb: Sha256Rnds2(dest, src2, xmm0); return true;
		case 0xcc: Sha256Msg1(dest, src2); return true;
		case 0xcd: Sha256Msg2(dest, src2); return true;
		default: return false;
	}
}

// Keep instruction semantics shared; only access to the saved host context differs.
struct Context {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	PCONTEXT native;

	[[nodiscard]] uint64_t Rip() const { return native->Rip; }
	void                   Advance(size_t length) { native->Rip += length; }
	[[nodiscard]] void*    Xmm(uint8_t index) const { return &native->Xmm0 + index; }

	uint64_t* GprPtr(uint8_t index) {
		switch (index) {
			case 0: return &native->Rax;
			case 1: return &native->Rcx;
			case 2: return &native->Rdx;
			case 3: return &native->Rbx;
			case 4: return &native->Rsp;
			case 5: return &native->Rbp;
			case 6: return &native->Rsi;
			case 7: return &native->Rdi;
			case 8: return &native->R8;
			case 9: return &native->R9;
			case 10: return &native->R10;
			case 11: return &native->R11;
			case 12: return &native->R12;
			case 13: return &native->R13;
			case 14: return &native->R14;
			case 15: return &native->R15;
			default: return nullptr;
		}
	}

	[[nodiscard]] uint32_t GetEflags() const { return native->EFlags; }
	void                   SetEflags(uint32_t flags) { native->EFlags = flags; }

	void LoadGprs(uint64_t (&gpr)[16]) const {
		const uint64_t registers[] = {native->Rax, native->Rcx, native->Rdx, native->Rbx,
		                              native->Rsp, native->Rbp, native->Rsi, native->Rdi,
		                              native->R8,  native->R9,  native->R10, native->R11,
		                              native->R12, native->R13, native->R14, native->R15};
		std::memcpy(gpr, registers, sizeof(gpr));
	}

	void ClearUpperYmm(uint8_t index) const {
		if ((native->ContextFlags & CONTEXT_XSTATE) != CONTEXT_XSTATE) {
			return;
		}
		DWORD64 features = 0;
		if (!GetXStateFeaturesMask(native, &features) || (features & XSTATE_MASK_AVX) == 0) {
			return; // An absent AVX component restores zeroes.
		}
		DWORD size = 0;
		auto* ymm = static_cast<M128A*>(LocateXStateFeature(native, XSTATE_AVX, &size));
		if (ymm != nullptr && size >= (index + 1u) * sizeof(M128A)) {
			ymm[index] = {};
		}
	}
#elif defined(__APPLE__)
	ucontext_t* native;

	[[nodiscard]] uint64_t Rip() const {
		return static_cast<uint64_t>(native->uc_mcontext->__ss.__rip);
	}
	void Advance(size_t length) {
		native->uc_mcontext->__ss.__rip += static_cast<uint64_t>(length);
	}
	// Darwin names the XMM file __fpu_xmm0..__fpu_xmm15 instead of exposing an array.
	[[nodiscard]] void* Xmm(uint8_t index) const {
		auto* fs = &native->uc_mcontext->__fs;
		switch (index) {
			case 0: return &fs->__fpu_xmm0;
			case 1: return &fs->__fpu_xmm1;
			case 2: return &fs->__fpu_xmm2;
			case 3: return &fs->__fpu_xmm3;
			case 4: return &fs->__fpu_xmm4;
			case 5: return &fs->__fpu_xmm5;
			case 6: return &fs->__fpu_xmm6;
			case 7: return &fs->__fpu_xmm7;
			case 8: return &fs->__fpu_xmm8;
			case 9: return &fs->__fpu_xmm9;
			case 10: return &fs->__fpu_xmm10;
			case 11: return &fs->__fpu_xmm11;
			case 12: return &fs->__fpu_xmm12;
			case 13: return &fs->__fpu_xmm13;
			case 14: return &fs->__fpu_xmm14;
			case 15: return &fs->__fpu_xmm15;
			default: return nullptr;
		}
	}

	uint64_t* GprPtr(uint8_t index) {
		switch (index) {
			case 0: return reinterpret_cast<uint64_t*>(&native->uc_mcontext->__ss.__rax);
			case 1: return reinterpret_cast<uint64_t*>(&native->uc_mcontext->__ss.__rcx);
			case 2: return reinterpret_cast<uint64_t*>(&native->uc_mcontext->__ss.__rdx);
			case 3: return reinterpret_cast<uint64_t*>(&native->uc_mcontext->__ss.__rbx);
			case 4: return reinterpret_cast<uint64_t*>(&native->uc_mcontext->__ss.__rsp);
			case 5: return reinterpret_cast<uint64_t*>(&native->uc_mcontext->__ss.__rbp);
			case 6: return reinterpret_cast<uint64_t*>(&native->uc_mcontext->__ss.__rsi);
			case 7: return reinterpret_cast<uint64_t*>(&native->uc_mcontext->__ss.__rdi);
			case 8: return reinterpret_cast<uint64_t*>(&native->uc_mcontext->__ss.__r8);
			case 9: return reinterpret_cast<uint64_t*>(&native->uc_mcontext->__ss.__r9);
			case 10: return reinterpret_cast<uint64_t*>(&native->uc_mcontext->__ss.__r10);
			case 11: return reinterpret_cast<uint64_t*>(&native->uc_mcontext->__ss.__r11);
			case 12: return reinterpret_cast<uint64_t*>(&native->uc_mcontext->__ss.__r12);
			case 13: return reinterpret_cast<uint64_t*>(&native->uc_mcontext->__ss.__r13);
			case 14: return reinterpret_cast<uint64_t*>(&native->uc_mcontext->__ss.__r14);
			case 15: return reinterpret_cast<uint64_t*>(&native->uc_mcontext->__ss.__r15);
			default: return nullptr;
		}
	}

	[[nodiscard]] uint32_t GetEflags() const {
		return static_cast<uint32_t>(native->uc_mcontext->__ss.__rflags);
	}
	void SetEflags(uint32_t flags) {
		native->uc_mcontext->__ss.__rflags = static_cast<uint64_t>(flags);
	}
#else
	ucontext_t* native;

	[[nodiscard]] uint64_t Rip() const {
		return static_cast<uint64_t>(native->uc_mcontext.gregs[REG_RIP]);
	}
	void Advance(size_t length) {
		native->uc_mcontext.gregs[REG_RIP] += static_cast<greg_t>(length);
	}
	[[nodiscard]] void* Xmm(uint8_t index) const {
		if (native->uc_mcontext.fpregs == nullptr) {
			return nullptr;
		}
		return native->uc_mcontext.fpregs->_xmm[index].element;
	}

	uint64_t* GprPtr(uint8_t index) {
		constexpr int registers[] = {REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP,
		                             REG_RSI, REG_RDI, REG_R8,  REG_R9,  REG_R10, REG_R11,
		                             REG_R12, REG_R13, REG_R14, REG_R15};
		if (index < 16) {
			return reinterpret_cast<uint64_t*>(&native->uc_mcontext.gregs[registers[index]]);
		}
		return nullptr;
	}

	[[nodiscard]] uint32_t GetEflags() const {
		return static_cast<uint32_t>(native->uc_mcontext.gregs[REG_EFL]);
	}
	void SetEflags(uint32_t flags) {
		native->uc_mcontext.gregs[REG_EFL] = static_cast<greg_t>(flags);
	}

	void LoadGprs(uint64_t (&gpr)[16]) const {
		constexpr int registers[] = {REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP,
		                             REG_RSI, REG_RDI, REG_R8,  REG_R9,  REG_R10, REG_R11,
		                             REG_R12, REG_R13, REG_R14, REG_R15};
		for (size_t i = 0; i < 16; ++i) {
			gpr[i] = static_cast<uint64_t>(native->uc_mcontext.gregs[registers[i]]);
		}
	}

	void ClearUpperYmm(uint8_t index) const {
		// Linux signal frames use the standard XSAVE layout. An absent AVX component
		// already restores the architectural initial value (all zeroes).
		auto*    state    = reinterpret_cast<uint8_t*>(native->uc_mcontext.fpregs);
		uint32_t magic    = 0;
		uint32_t size     = 0;
		uint64_t features = 0;
		std::memcpy(&magic, state + 464, sizeof(magic));
		if (magic != 0x46505853) {
			return;
		}
		std::memcpy(&size, state + 480, sizeof(size));
		if (size < 832) {
			return;
		}
		std::memcpy(&features, state + 512, sizeof(features));
		if ((features & 4) != 0) {
			std::memset(state + 576 + index * 16, 0, 16);
		}
	}
#endif
};

#if !defined(__APPLE__)

static bool TryEmulateShaNi(Context& context) {
	const auto* rip = reinterpret_cast<const uint8_t*>(context.Rip());
	ShaNiInsn   insn {};
	if (!DecodeShaNiInsn(rip, insn)) {
		return false;
	}

	const uint8_t modrm_byte = rip[insn.modrm_offset];
	const uint8_t dest_index = ShaNiRegIndex(modrm_byte, insn.rex, true);
	auto*         dest_xmm   = context.Xmm(dest_index);
	auto*         xmm0       = context.Xmm(0);
	if (dest_xmm == nullptr || xmm0 == nullptr) {
		return false;
	}

	XmmWords dest {};
	XmmWords src2 {};
	XmmWords xmm0_words {};
	std::memcpy(&dest, dest_xmm, sizeof(dest));
	std::memcpy(&xmm0_words, xmm0, sizeof(xmm0_words));

	if (ShaNiModrmIsRegister(modrm_byte)) {
		const uint8_t src_index = ShaNiRegIndex(modrm_byte, insn.rex, false);
		auto*         src_xmm   = context.Xmm(src_index);
		if (src_xmm == nullptr) {
			return false;
		}
		std::memcpy(&src2, src_xmm, sizeof(src2));
	} else {
		uint64_t    gpr[16] {};
		const void* source = nullptr;
		context.LoadGprs(gpr);
		if (!ResolveShaNiMemoryAddress(rip, insn, gpr, source)) {
			return false;
		}
		std::memcpy(&src2, source, sizeof(src2));
	}

	if (!ExecuteShaNiInsn(insn, src2, xmm0_words, dest)) {
		return false;
	}

	std::memcpy(dest_xmm, &dest, sizeof(dest));
	context.Advance(insn.length);
	return true;
}

#endif

static bool TryEmulateSse4a(Context& context) {
	const auto*   rip    = reinterpret_cast<const uint8_t*>(context.Rip());
	const uint8_t prefix = rip[0];
	if (prefix != 0x66 && prefix != 0xf2) {
		return false;
	}

	size_t  offset = 1;
	uint8_t rex    = 0;
	if ((rip[offset] & 0xf0u) == 0x40u) {
		rex = rip[offset++];
	}
	if (rip[offset] != 0x0f) {
		return false;
	}
	const bool register_extract = prefix == 0x66 && rip[offset + 1] == 0x79;
	if (rip[offset + 1] != 0x78 && !register_extract) {
		return false;
	}

	const uint8_t modrm = rip[offset + 2];
	if ((modrm & 0xc0u) != 0xc0u) {
		return false;
	}

	const uint8_t reg = ((modrm >> 3u) & 0x07u) | ((rex & 0x04u) << 1u);
	const uint8_t rm  = (modrm & 0x07u) | ((rex & 0x01u) << 3u);

	// Immediate EXTRQ encodes its destination in r/m; the two-register form uses reg.
	uint8_t dest_index = reg;
	if (prefix == 0x66 && !register_extract) {
		dest_index = rm;
	}
	auto* dest_xmm = context.Xmm(dest_index);
	auto* src_xmm  = context.Xmm(rm);
	if (dest_xmm == nullptr || src_xmm == nullptr) {
		return false;
	}
	uint64_t dest[2] {};
	uint64_t source = 0;
	std::memcpy(dest, dest_xmm, sizeof(dest));
	std::memcpy(&source, src_xmm, sizeof(source));
	uint8_t length             = 0;
	uint8_t index              = 0;
	size_t  instruction_length = offset + 3;
	if (register_extract) {
		length = static_cast<uint8_t>(source);
		index  = static_cast<uint8_t>(source >> 8u);
	} else {
		length = rip[offset + 3];
		index  = rip[offset + 4];
		instruction_length += 2;
	}
	if (prefix == 0x66) {
		dest[0] = ExtractBitField(dest[0], length, index);
		dest[1] = 0;
	} else {
		dest[0] = InsertBitField(dest[0], source, length, index);
	}
	std::memcpy(dest_xmm, dest, sizeof(dest));
	context.Advance(instruction_length);
	return true;
}

#if !defined(__APPLE__)

static bool TryEmulateMonitorxMwaitx(Context& context) {
	const auto* rip = reinterpret_cast<const uint8_t*>(context.Rip());
	if (rip[0] != 0x0f || rip[1] != 0x01 || (rip[2] != 0xfa && rip[2] != 0xfb)) {
		return false;
	}

	// Approximate AMD MONITORX/MWAITX as no-op/yield.
	if (rip[2] == 0xfb) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SwitchToThread();
#else
		::sched_yield();
#endif
	}
	context.Advance(3);
	return true;
}

static uint32_t ReciprocalSquareRoot(uint32_t bits) {
	const uint32_t magnitude = bits & 0x7fffffffu;
	const uint32_t exponent  = magnitude & 0x7f800000u;
	if (exponent == 0) {
		// RSQRT treats denormals as signed zero regardless of MXCSR.DAZ.
		return (bits & 0x80000000u) | 0x7f800000u;
	}
	if (magnitude > 0x7f800000u) {
		return bits | 0x00400000u; // Quiet NaNs without raising an exception.
	}
	if ((bits & 0x80000000u) != 0) {
		return 0xffc00000u;
	}
	if (magnitude == 0x7f800000u) {
		return 0;
	}

	// A deterministic accurate estimate meets the instruction's relative-error
	// bound without relying on the host vendor's approximation table.
	const __m128d input  = _mm_set_sd(static_cast<double>(std::bit_cast<float>(bits)));
	const __m128d result = _mm_div_sd(_mm_set_sd(1.0), _mm_sqrt_sd(input, input));
	return std::bit_cast<uint32_t>(_mm_cvtss_f32(_mm_cvtsd_ss(_mm_setzero_ps(), result)));
}

static bool TryEmulateReciprocalSquareRoot(Context& context) {
	const auto* rip            = reinterpret_cast<const uint8_t*>(context.Rip());
	size_t      prefix_size    = 0;
	uint8_t     dest_extension = 0;
	uint8_t     src_extension  = 0;
	if (rip[0] == 0xc5 && (rip[1] & 0x7fu) == 0x70u) {
		prefix_size    = 2;
		dest_extension = (~rip[1] & 0x80u) >> 4u;
	} else if (rip[0] == 0xc4 && (rip[1] & 0x1fu) == 1 && (rip[2] & 0x7fu) == 0x70u) {
		prefix_size    = 3;
		dest_extension = (~rip[1] & 0x80u) >> 4u;
		src_extension  = (~rip[1] & 0x20u) >> 2u;
	} else {
		return false;
	}
	if (rip[prefix_size] != 0x52 || (rip[prefix_size + 1] & 0xc0u) != 0xc0u) {
		return false;
	}

	const uint8_t modrm    = rip[prefix_size + 1];
	const uint8_t dest     = ((modrm >> 3u) & 7u) | dest_extension;
	const uint8_t source   = (modrm & 7u) | src_extension;
	auto*         dest_xmm = context.Xmm(dest);
	auto*         src_xmm  = context.Xmm(source);
	if (dest_xmm == nullptr || src_xmm == nullptr) {
		return false;
	}
	XmmWords result {};
	std::memcpy(&result, src_xmm, sizeof(result));
	// RSQRT ignores the rounding mode and never changes guest exception flags.
	// Mask host exceptions while calculating, then restore the handler's state.
	const uint32_t mxcsr = _mm_getcsr();
	_mm_setcsr(0x1f80);
	for (auto& word: result.w) {
		word = ReciprocalSquareRoot(word);
	}
	_mm_setcsr(mxcsr);
	std::memcpy(dest_xmm, &result, sizeof(result));
	context.ClearUpperYmm(dest);
	context.Advance(prefix_size + 2);
	return true;
}

#endif

bool IsReciprocalSquareRoot(const ZydisDecodedInstruction& instruction,
                            const ZydisDecodedOperand* operands) {
	return instruction.mnemonic == ZYDIS_MNEMONIC_VRSQRTPS &&
	       instruction.encoding == ZYDIS_INSTRUCTION_ENCODING_VEX &&
	       instruction.raw.vex.offset == 0 && operands[0].size == 128 &&
	       operands[1].type == ZYDIS_OPERAND_TYPE_REGISTER;
}

uint64_t PatchReciprocalSquareRoots(uint64_t address, uint64_t size) {
	uint64_t patched = 0;
#if !defined(__APPLE__)
	ZydisDecoder decoder {};
	if (!ZYAN_SUCCESS(
	        ZydisDecoderInit(&decoder, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64))) {
		return 0;
	}
	for (uint64_t offset = 0; offset < size;) {
		auto*                   code = reinterpret_cast<uint8_t*>(address + offset);
		ZydisDecodedInstruction instruction {};
		ZydisDecodedOperand     operands[ZYDIS_MAX_OPERAND_COUNT] {};
		if (!ZYAN_SUCCESS(
		        ZydisDecoderDecodeFull(&decoder, code, size - offset, &instruction, operands))) {
			++offset;
			continue;
		}
		if (IsReciprocalSquareRoot(instruction, operands)) {
			// vvvv is reserved (must be 1111b). Clear one bit to route this
			// otherwise intact instruction through the illegal-instruction emulator.
			code[instruction.raw.vex.size - 1] &= ~0x08u;
			++patched;
		}
		offset += instruction.length;
	}
#else
	(void)address;
	(void)size;
#endif
	return patched;
}

bool TryEmulate(void* native_context) {
	if (native_context == nullptr) {
		return false;
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	Context context {static_cast<PCONTEXT>(native_context)};
#elif defined(__APPLE__)
	auto* saved_context = static_cast<ucontext_t*>(native_context);
	if (saved_context->uc_mcontext == nullptr) {
		return false;
	}
	Context context {saved_context};
#else
	Context context {static_cast<ucontext_t*>(native_context)};
#endif
#if !defined(__APPLE__)
	if (TryEmulateReciprocalSquareRoot(context)) {
		return true;
	}
	return TryEmulateMonitorxMwaitx(context) || TryEmulateSse4a(context) ||
	       TryEmulateShaNi(context);
#else
	return TryEmulateSse4a(context);
#endif
}

static void ClearRegister(Context& context, ZydisRegister reg,
                          const ZydisDecodedInstruction& instruction,
                          const ZydisDecodedOperand& operand) {
	const auto reg_class = ZydisRegisterGetClass(reg);

	if (reg_class == ZYDIS_REGCLASS_GPR) {
		if (reg == ZYDIS_REGISTER_AH) {
			if (auto* ptr = context.GprPtr(0)) {
				*ptr &= ~0xFF00ull;
			}
			return;
		}
		if (reg == ZYDIS_REGISTER_CH) {
			if (auto* ptr = context.GprPtr(1)) {
				*ptr &= ~0xFF00ull;
			}
			return;
		}
		if (reg == ZYDIS_REGISTER_DH) {
			if (auto* ptr = context.GprPtr(2)) {
				*ptr &= ~0xFF00ull;
			}
			return;
		}
		if (reg == ZYDIS_REGISTER_BH) {
			if (auto* ptr = context.GprPtr(3)) {
				*ptr &= ~0xFF00ull;
			}
			return;
		}

		const int reg_id = ZydisRegisterGetId(reg);
		if (reg_id < 0 || reg_id >= 16) {
			return;
		}
		auto* ptr = context.GprPtr(static_cast<uint8_t>(reg_id));
		if (ptr == nullptr) {
			return;
		}

		if (operand.size >= 32) {
			// In x86-64, writes to 32-bit or 64-bit GPRs zero the upper 32/64 bits.
			*ptr = 0;
		} else if (operand.size == 16) {
			*ptr &= ~0xFFFFull;
		} else if (operand.size == 8) {
			*ptr &= ~0xFFull;
		}
		return;
	}

	if (reg_class == ZYDIS_REGCLASS_XMM) {
		const int reg_id = ZydisRegisterGetId(reg);
		if (reg_id >= 0 && reg_id < 16) {
			if (auto* xmm = context.Xmm(static_cast<uint8_t>(reg_id))) {
				std::memset(xmm, 0, 16);
			}
			if (instruction.encoding == ZYDIS_INSTRUCTION_ENCODING_VEX) {
				context.ClearUpperYmm(static_cast<uint8_t>(reg_id));
			}
		}
		return;
	}

	if (reg_class == ZYDIS_REGCLASS_YMM) {
		const int reg_id = ZydisRegisterGetId(reg);
		if (reg_id >= 0 && reg_id < 16) {
			if (auto* xmm = context.Xmm(static_cast<uint8_t>(reg_id))) {
				std::memset(xmm, 0, 16);
			}
			context.ClearUpperYmm(static_cast<uint8_t>(reg_id));
		}
		return;
	}
}

bool TryRecoverAccessViolation(void* native_context, uint64_t fault_addr, bool is_write) {
	if (native_context == nullptr) {
		return false;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	Context context {static_cast<PCONTEXT>(native_context)};
#elif defined(__APPLE__)
	auto* saved_context = static_cast<ucontext_t*>(native_context);
	if (saved_context->uc_mcontext == nullptr) {
		return false;
	}
	Context context {saved_context};
#else
	Context context {static_cast<ucontext_t*>(native_context)};
#endif

	const uint64_t rip = context.Rip();
	if (rip == 0) {
		return false;
	}

	// Prevent infinite fault loop at the exact same instruction pointer
	thread_local uint64_t t_last_rip           = 0;
	thread_local uint32_t t_consecutive_faults = 0;
	if (rip == t_last_rip) {
		if (++t_consecutive_faults > 100) {
			return false;
		}
	} else {
		t_last_rip           = rip;
		t_consecutive_faults = 1;
	}

	ZydisDecoder decoder {};
	if (!ZYAN_SUCCESS(
	        ZydisDecoderInit(&decoder, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64))) {
		return false;
	}

	ZydisDecodedInstruction instruction {};
	ZydisDecodedOperand     operands[ZYDIS_MAX_OPERAND_COUNT] {};
	if (!ZYAN_SUCCESS(ZydisDecoderDecodeFull(
	        &decoder, reinterpret_cast<const void*>(rip), 15, &instruction, operands))) {
		return false;
	}

	if (instruction.length == 0 || instruction.length > 15) {
		return false;
	}

	// Stack push: decrement RSP
	if (instruction.mnemonic == ZYDIS_MNEMONIC_PUSH) {
		auto* rsp = context.GprPtr(4);
		if (rsp != nullptr) {
			*rsp -= 8;
		}
		context.Advance(instruction.length);
		return true;
	}

	// Stack pop: clear destination register and increment RSP
	if (instruction.mnemonic == ZYDIS_MNEMONIC_POP) {
		if (instruction.operand_count > 0 && operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
			ClearRegister(context, operands[0].reg.value, instruction, operands[0]);
		}
		auto* rsp = context.GprPtr(4);
		if (rsp != nullptr) {
			*rsp += 8;
		}
		context.Advance(instruction.length);
		return true;
	}

	// TEST: 0 & anything = 0 -> ZF=1, PF=1; clear CF, SF, OF
	if (instruction.mnemonic == ZYDIS_MNEMONIC_TEST) {
		uint32_t flags = context.GetEflags();
		flags &= ~(0x0001u | 0x0004u | 0x0010u | 0x0040u | 0x0080u | 0x0800u);
		flags |= (0x0040u | 0x0004u);
		context.SetEflags(flags);
		context.Advance(instruction.length);
		return true;
	}

	// CMP: assume memory value is 0 -> ZF=1, PF=1
	if (instruction.mnemonic == ZYDIS_MNEMONIC_CMP) {
		uint32_t flags = context.GetEflags();
		flags &= ~(0x0001u | 0x0004u | 0x0010u | 0x0040u | 0x0080u | 0x0800u);
		flags |= (0x0040u | 0x0004u);
		context.SetEflags(flags);
		context.Advance(instruction.length);
		return true;
	}

	// Check if this instruction accesses memory
	bool accesses_memory = false;
	for (uint8_t i = 0; i < instruction.operand_count; ++i) {
		if (operands[i].type == ZYDIS_OPERAND_TYPE_MEMORY) {
			accesses_memory = true;
			break;
		}
	}

	if (!accesses_memory) {
		return false;
	}

	// If the instruction writes to register(s) from memory, clear them to neutral zero.
	// For memory writes, the write to invalid memory is dropped (no-op).
	for (uint8_t i = 0; i < instruction.operand_count; ++i) {
		const auto& op = operands[i];
		if (op.type == ZYDIS_OPERAND_TYPE_REGISTER &&
		    (op.actions & ZYDIS_OPERAND_ACTION_WRITE) != 0) {
			if (op.reg.value != ZYDIS_REGISTER_RSP &&
			    op.reg.value != ZYDIS_REGISTER_RFLAGS) {
				ClearRegister(context, op.reg.value, instruction, op);
			}
		}
	}

	static std::atomic<uint64_t> s_recovered_count {0};
	const uint64_t count = s_recovered_count.fetch_add(1, std::memory_order_relaxed);
	if (count < 20 || (count % 500) == 0) {
		std::printf("[MemoryRecovery] Intercepted and recovered guest access violation #%" PRIu64
		            " at RIP=0x%016" PRIx64 " (len=%u, addr=0x%016" PRIx64 ", %s)\n",
		            count + 1, rip, instruction.length, fault_addr, is_write ? "write" : "read");
		std::fflush(stdout);
	}

	context.Advance(instruction.length);
	return true;
}

} // namespace Loader::X64InstructionEmulator
