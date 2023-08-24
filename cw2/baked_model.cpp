#include "baked_model.hpp"

#include <cstdio>
#include <cstring>
#include <glm/gtc/quaternion.hpp>
#include "../labutils/error.hpp"
#include "compute.h"
namespace lut = labutils;

namespace
{
	// See cw2-bake/main.cpp for more info
	constexpr char kFileMagic[16] = "\0\0COMP5822Mmesh";
	constexpr char kFileVariant[16] = "default";

	constexpr std::uint32_t kMaxString = 32*1024;

	// functions
	BakedModel load_baked_model_( FILE*, char const* );
}

BakedModel load_baked_model( char const* aModelPath )
{
	FILE* fin = std::fopen( aModelPath, "rb" );
	if( !fin )
		throw lut::Error( "load_baked_model(): unable to open '%s' for reading", aModelPath );

	try
	{
		auto ret = load_baked_model_( fin, aModelPath );
		std::fclose( fin );
		return ret;
	}
	catch( ... )
	{
		std::fclose( fin );
		throw;
	}
}

namespace
{
	void checked_read_( FILE* aFin, std::size_t aBytes, void* aBuffer )
	{
		auto ret = std::fread( aBuffer, 1, aBytes, aFin );

		if( aBytes != ret )
			throw lut::Error( "checked_read_(): expected %zu bytes, got %zu", aBytes, ret );
	}

	std::uint32_t read_uint32_( FILE* aFin )
	{
		std::uint32_t ret;
		checked_read_( aFin, sizeof(std::uint32_t), &ret );
		return ret;
	}
	std::string read_string_( FILE* aFin )
	{
		auto const length = read_uint32_( aFin );

		if( length >= kMaxString )
			throw lut::Error( "read_string_(): unexpectedly long string (%u bytes)", length );

		std::string ret;
		ret.resize( length );

		checked_read_( aFin, length, ret.data() );
		return ret;
	}

	float rsqrt(float x)
	{
		return 1/std::sqrt(x);
	}

	float remap(float x, float min1, float max1, float min2, float max2)
	{
		return (((x - min1) / (max1 - min1)) * (max2 - min2)) + min2;
	}

	 template<unsigned N>
	uint16_t  encode_unorm( float x )
	{
		return uint16_t( int (x * ((1<<(N))-1) + 0.5f) );
	}

	template<unsigned N>
	uint16_t  encode_snorm(float x)
	{
		return (x < 0) | (encode_unorm<N - 1>(x < 0 ? -x : x) << 1);
	}

	union FP32 {
		uint32_t u;
		float f;
		struct {
			uint32_t Mantissa : 23;
			uint32_t Exponent : 8;
			uint32_t Sign : 1;
		};
	};

	union FP16 {
		uint16_t u;
		struct {
			uint32_t Mantissa : 10;
			uint32_t Exponent : 5;
			uint32_t Sign : 1;
		};
	};

	uint16_t encode16_half(float fl) {
		FP16 o = { 0 };
		FP32 f; f.f = fl;
		// Based on ISPC reference code (with minor modifications)
		if (f.Exponent == 0) {
			// if Signed zero/denormal (which will underflow)
			o.Exponent = 0;
		}
		else if (f.Exponent == 255) {
			// if Inf or NaN (all exponent bits set)
			o.Exponent = 31;
			o.Mantissa = f.Mantissa ? 0x200 : 0; // NaN->qNaN and Inf->Inf
		}
		else {
			// if Normalized number
			// Exponent unbias the single, then bias the halfp
			int newexp = f.Exponent - 127 + 15;
			if (newexp >= 31) {
				// if Overflow, return signed infinity
				o.Exponent = 31;
			}
			else if (newexp <= 0) {
				// if Underflow
				if ((14 - newexp) <= 24) {
					// Mantissa might be non-zero
					uint32_t mant = f.Mantissa | 0x800000; // Hidden 1 bit
					o.Mantissa = mant >> (14 - newexp);
					// Check for rounding
					if ((mant >> (13 - newexp)) & 1) {
						// Round, might overflow into exp bit, but this is OK
						o.u++;
					}
				}
			}
			else {
				o.Exponent = newexp;
				o.Mantissa = f.Mantissa >> 13;
				// Check for rounding
				if (f.Mantissa & 0x1000) {
					// Round, might overflow to inf, this is OK
					o.u++;
				}
			}
		}
		o.Sign = f.Sign;
		return o.u;
	}

	template<unsigned N>
	uint16_t  encode_half( float fl )
	{
		return encode16_half(fl) >> (16-N);
	}


	void encode_quat(uint32_t &out,float x, float y, float z,float w)
	{
		const float rmin = -rsqrt(2), rmax = rsqrt(2);
		float x2 = x*x , y2 = y*y , z2 = z*z , w2 = w * w;
		if (x2 >= y2 && x2 >= z2 && x2 >= w2)
		{
			y = remap(y, rmin, rmax, -1, 1);
			z = remap(z, rmin, rmax, -1, 1);
			w = remap(w, rmin, rmax, -1, 1);
			out = x >= 0 ? static_cast<uint32_t>((0 << 30) | (encode_snorm<10>(y) << 20) | (encode_snorm<10>(z) << 10) | (encode_snorm<10>(w) << 0))
				: static_cast<uint32_t>((0 << 30) | (encode_snorm<10>(-y) << 20) | (encode_snorm<10>(-z) << 10) | (encode_snorm<10>(-w) << 0));
		}
		else if (y2 >= z2 && y2 >= w2)
		{
			x = remap(x, rmin, rmax, -1, 1);
			z = remap(z, rmin, rmax, -1, 1);
			w = remap(w, rmin, rmax, -1, 1);
			out = y >= 0 ? static_cast<uint32_t>((1 << 30) | (encode_snorm<10>(x) << 20) | (encode_snorm<10>(z) << 10) | (encode_snorm<10>(w) << 0))
				: static_cast<uint32_t>((1 << 30) | (encode_snorm<10>(-x) << 20) | (encode_snorm<10>(-z) << 10) | (encode_snorm<10>(-w) << 0));
		}
		else if(z2 >= w2)
		{
			x = remap(x, rmin, rmax, -1, 1);
			y = remap(y, rmin, rmax, -1, 1);
			w = remap(w, rmin, rmax, -1, 1);
			out = y >= 0 ? static_cast<uint32_t>((2 << 30) | (encode_snorm<10>(x) << 20) | (encode_snorm<10>(y) << 10) | (encode_snorm<10>(w) << 0))
				: static_cast<uint32_t>((2 << 30) | (encode_snorm<10>(-x) << 20) | (encode_snorm<10>(-y) << 10) | (encode_snorm<10>(-w) << 0));
		}
		else
		{
			x = remap(x, rmin, rmax, -1, 1);
			y = remap(y, rmin, rmax, -1, 1);
			z = remap(z, rmin, rmax, -1, 1);
			out = y >= 0 ? static_cast<uint32_t>((3 << 30) | (encode_snorm<10>(x) << 20) | (encode_snorm<10>(y) << 10) | (encode_snorm<10>(z) << 0))
				: static_cast<uint32_t>((3 << 30) | (encode_snorm<10>(-x) << 20) | (encode_snorm<10>(-y) << 10) | (encode_snorm<10>(-z) << 0));
		}
	}


	BakedModel load_baked_model_( FILE* aFin, char const* aInputName )
	{
		BakedModel ret;

		// Figure out base path
		char const* pathBeg = aInputName;
		char const* pathEnd = std::strrchr( pathBeg, '/' );
	
		std::string const prefix = pathEnd
			? std::string( pathBeg, pathEnd+1 )
			: ""
		;

		// Read header and verify file magic and variant
		char magic[16];
		checked_read_( aFin, 16, magic );

		if( 0 != std::memcmp( magic, kFileMagic, 16 ) )
			throw lut::Error( "load_baked_model_(): %s: invalid file signature!", aInputName );

		char variant[16];
		checked_read_( aFin, 16, variant );

		if( 0 != std::memcmp( variant, kFileVariant, 16 ) )
			throw lut::Error( "load_baked_model_(): %s: file variant is '%s', expected '%s'", aInputName, variant, kFileVariant );

		// Read texture info
		auto const textureCount = read_uint32_( aFin );
		for( std::uint32_t i = 0; i < textureCount; ++i )
		{
			BakedTextureInfo info;
			info.path = prefix + read_string_( aFin );

			std::uint8_t channels;
			checked_read_( aFin, sizeof(std::uint8_t), &channels );
			info.channels = channels;

			ret.textures.emplace_back( std::move(info) );
		}

		// Read material info
		auto const materialCount = read_uint32_( aFin );
		for( std::uint32_t i = 0; i < materialCount; ++i )
		{
			BakedMaterialInfo info;
			info.baseColorTextureId = read_uint32_( aFin );
			info.roughnessTextureId = read_uint32_( aFin );
			info.metalnessTextureId = read_uint32_( aFin );
			info.alphaMaskTextureId = read_uint32_( aFin );
			info.normalMapTextureId = read_uint32_( aFin );

			assert( info.baseColorTextureId < ret.textures.size() );
			assert( info.roughnessTextureId < ret.textures.size() );
			assert( info.metalnessTextureId < ret.textures.size() );

			ret.materials.emplace_back( std::move(info) );
		}

		// Read mesh data
		auto const meshCount = read_uint32_( aFin );
		for( std::uint32_t i = 0; i < meshCount; ++i )
		{
			BakedMeshData data;
			data.materialId = read_uint32_( aFin );
			assert( data.materialId < ret.materials.size() );

			auto const V = read_uint32_( aFin );
			auto const I = read_uint32_( aFin );

			data.positions.resize( V );
			checked_read_( aFin, V*sizeof(glm::vec3), data.positions.data() );

			data.normals.resize( V );
			checked_read_( aFin, V*sizeof(glm::vec3), data.normals.data() );

			data.texcoords.resize( V );
			checked_read_( aFin, V*sizeof(glm::vec2), data.texcoords.data() );

			data.indices.resize( I );
			checked_read_( aFin, I*sizeof(std::uint32_t), data.indices.data() );

			data.tangents.resize(V);

			std::vector<double> positions3D(data.positions.size() * 3);

			for (std::size_t i = 0; i < data.positions.size(); ++i) {
				positions3D[i * 3 + 0] = data.positions[i].x;
				positions3D[i * 3 + 1] = data.positions[i].y;
				positions3D[i * 3 + 2] = data.positions[i].z;
			}

			std::vector<double> uv2D(data.texcoords.size() * 2);

			for (std::size_t i = 0; i < data.positions.size(); ++i) {
				uv2D[i * 2 + 0] = data.texcoords[i].x;
				uv2D[i * 2 + 1] = data.texcoords[i].y;
			}

			std::vector<double> normals3D(data.positions.size() * 3);

			for (std::size_t i = 0; i < data.positions.size(); ++i) {
				normals3D[i * 3 + 0] = data.normals[i].x;
				normals3D[i * 3 + 1] = data.normals[i].y;
				normals3D[i * 3 + 2] = data.normals[i].z;
			}

			std::vector<double> ctangents3d(V * 3);
			std::vector<double> cbitangents3d(V * 3);
			std::vector<double> tangents3d(V * 3);
			std::vector<double> bitangents3d(V * 3);
			std::vector<double> tangents4d(V * 4);

			compute::computeCornerTSpace(
				data.indices,
				data.indices,
				positions3D,
				uv2D,
				ctangents3d,
				cbitangents3d
			);

			compute::computeVertexTSpace(
				data.indices,
				ctangents3d,
				cbitangents3d,
				V,
				tangents3d,
				bitangents3d
			);

			compute::orthogonalizeTSpace(
				normals3D,
				tangents3d,
				bitangents3d
			);

			compute::computeTangent4D(
				normals3D,
				tangents3d,
				bitangents3d,
				tangents4d
			);

			std::vector<glm::vec4> tangents(V);

			for (std::size_t i = 0; i < V; ++i) {
				tangents[i] = glm::vec4(tangents4d[i * 4], tangents4d[i * 4 + 1],
					tangents4d[i * 4 + 2], tangents4d[i * 4 + 3]);
			}

			data.tangents = tangents;

			std::vector<glm::vec3> bitangents;
			for (size_t i = 0; i < tangents.size(); ++i) {
				glm::vec3 bitangent = abs(glm::normalize(glm::cross(data.normals[i], glm::vec3(tangents[i]))));
				bitangents.push_back(bitangent);
			}

			std::vector<glm::quat> tbnQuaternions;
			tbnQuaternions.reserve(data.normals.size());

			for (size_t i = 0; i < data.normals.size(); ++i) {
				glm::mat3 tbnMatrix(tangents[i], bitangents[i], data.normals[i]);
				glm::quat tbnQuaternion = glm::normalize(glm::quat_cast(tbnMatrix));
				tbnQuaternions.push_back(tbnQuaternion);
			}

			std::vector<glm::uint32> encodedtbnQuaternions;
			encodedtbnQuaternions.resize(data.normals.size());
			for (size_t i = 0; i < data.normals.size(); ++i) {
				 encode_quat(encodedtbnQuaternions[i],tbnQuaternions[i].x, tbnQuaternions[i].y, tbnQuaternions[i].z, tbnQuaternions[i].w );
			}

			//tangents
			data.packedTBN = encodedtbnQuaternions;

			ret.meshes.emplace_back( std::move(data) );
		}

		// Check
		char byte;
		auto const check = std::fread( &byte, 1, 1, aFin );
		
		if( 0 != check )
			std::fprintf( stderr, "Note: '%s' contains trailing bytes\n", aInputName );

		return ret;
	}
}
