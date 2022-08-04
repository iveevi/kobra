#include <fstream>
#include <map>
#include <string>

#include <sutil/sutil.h>
#include <sutil/PPMLoader.h>
#include <sutil/vec_math.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

/*#define TINYEXR_IMPLEMENTATION
#include <tinyexr/tinyexr.h> */

#include "exception.h"
#include "color.cuh"

namespace sutil {

static void savePPM( const unsigned char* Pix, const char* fname, int wid, int hgt, int chan )
{
    if( Pix == NULL || wid < 1 || hgt < 1 )
        throw Exception( "savePPM: Image is ill-formed. Not saving" );

    if( chan != 1 && chan != 3 && chan != 4 )
        throw Exception( "savePPM: Attempting to save image with channel count != 1, 3, or 4." );

    std::ofstream OutFile( fname, std::ios::out | std::ios::binary );
    if( !OutFile.is_open() )
        throw Exception( "savePPM: Could not open file for" );

    bool is_float = false;
    OutFile << 'P';
    OutFile << ( ( chan == 1 ? ( is_float ? 'Z' : '5' ) : ( chan == 3 ? ( is_float ? '7' : '6' ) : '8' ) ) )
            << std::endl;
    OutFile << wid << " " << hgt << std::endl << 255 << std::endl;

    OutFile.write( reinterpret_cast<char*>( const_cast<unsigned char*>( Pix ) ), wid*hgt*chan*( is_float ? 4 : 1 ) );
    OutFile.close();
}

// Returns string of file extension including '.'
static std::string fileExtensionForLoading()
{
    std::string extension;
#if SAMPLES_INPUT_GENERATE_PTX
    extension = ".ptx";
#endif
#if SAMPLES_INPUT_GENERATE_OPTIXIR
    extension = ".optixir";
#endif
    if( const char* ext = getenv("OPTIX_SAMPLES_INPUT_EXTENSION") )
    {
        extension = ext;
        if( extension.size() && extension[0] != '.' )
            extension = "." + extension;
    }
    return extension;
}

static bool fileExists( const char* path )
{
    std::ifstream str( path );
    return static_cast<bool>( str );
}

static bool fileExists( const std::string& path )
{
    return fileExists( path.c_str() );
}

static bool readSourceFile( std::string& str, const std::string& filename )
{
    // Try to open file
    std::ifstream file( filename.c_str(), std::ios::binary );
    if( file.good() )
    {
        // Found usable source file
        std::vector<unsigned char> buffer = std::vector<unsigned char>( std::istreambuf_iterator<char>( file ), {} );
        str.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}

static std::string sampleInputFilePath( const char* sampleName, const char* fileName )
{
    // Allow for overrides.
    static const char* directories[] =
    {
        // TODO: Remove the environment variable OPTIX_EXP_SAMPLES_SDK_PTX_DIR once SDK 6/7 packages are split
        getenv( "OPTIX_EXP_SAMPLES_SDK_PTX_DIR" ),
        getenv( "OPTIX_SAMPLES_SDK_PTX_DIR" ),
 #if defined(CMAKE_INTDIR)
        SAMPLES_PTX_DIR "/" CMAKE_INTDIR,
#endif
        SAMPLES_PTX_DIR,
        "."
    };

    // Allow overriding the file extension
    std::string extension = fileExtensionForLoading();

    if( !sampleName )
        sampleName = "sutil";
    for( const char* directory : directories )
    {
        if( directory )
        {
            std::string path = directory;
            path += '/';
            path += sampleName;
            path += "_generated_";
            path += fileName;
            path += extension;
            if( fileExists( path ) )
                return path;
        }
    }

    std::string error = "sutil::samplePTXFilePath couldn't locate ";
    error += fileName;
    error += " for sample ";
    error += sampleName;
    throw Exception( error.c_str() );
}

static void getInputDataFromFile( std::string& ptx, const char* sample_name, const char* filename )
{
    const std::string sourceFilePath = sampleInputFilePath( sample_name, filename );

    // Try to open source PTX file
    if( !readSourceFile( ptx, sourceFilePath ) )
    {
        std::string err = "Couldn't open source file " + sourceFilePath;
        throw std::runtime_error( err.c_str() );
    }
}

struct PtxSourceCache
{
    std::map<std::string, std::string*> map;
    ~PtxSourceCache()
    {
        for( std::map<std::string, std::string*>::const_iterator it = map.begin(); it != map.end(); ++it )
            delete it->second;
    }
};
static PtxSourceCache g_ptxSourceCache;

const char* getInputData( const char*                     sample,
                          const char*                     sampleDir,
                          const char*                     filename,
                          size_t&                         dataSize,
                          const char**                    log,
                          const std::vector<const char*>& compilerOptions )
{
    if( log )
        *log = NULL;

    std::string *                                 ptx, cu;
    std::string                                   key  = std::string( filename ) + ";" + ( sample ? sample : "" );
    std::map<std::string, std::string*>::iterator elem = g_ptxSourceCache.map.find( key );

    if( elem == g_ptxSourceCache.map.end() )
    {
        ptx = new std::string();
#if CUDA_NVRTC_ENABLED
        SUTIL_ASSERT( fileExtensionForLoading() == ".ptx" );
        std::string location;
        getCuStringFromFile( cu, location, sampleDir, filename );
        getPtxFromCuString( *ptx, sampleDir, cu.c_str(), location.c_str(), log, compilerOptions );
#else
        getInputDataFromFile( *ptx, sample, filename );
#endif
        g_ptxSourceCache.map[key] = ptx;
    }
    else
    {
        ptx = elem->second;
    }
    dataSize = ptx->size();
    return ptx->c_str();
}

void saveImage( const char* fname, const ImageBuffer& image, bool disable_srgb_conversion )
{
    const std::string filename( fname );
    if( filename.length() < 5 )
        throw Exception( "sutil::saveImage(): Failed to determine filename extension" );

    const std::string ext = filename.substr( filename.length()-3 );
    /* if( ext == "PPM" || ext == "ppm" )
    {
        //
        // Note -- we are flipping image vertically as we write it into output buffer
        //
        const int32_t width  = image.width;
        const int32_t height = image.height;
        std::vector<unsigned char> pix( width*height*3 );

        switch( image.pixel_format )
        {
            case BufferImageFormat::UNSIGNED_BYTE4:
            {
                for( int j = height - 1; j >= 0; --j )
                {
                    for( int i = 0; i < width; ++i )
                    {
                        const int32_t dst_idx = 3*width*(height-j-1) + 3*i;
                        const int32_t src_idx = 4*width*j            + 4*i;
                        pix[ dst_idx+0] = reinterpret_cast<uint8_t*>( image.data )[ src_idx+0 ];
                        pix[ dst_idx+1] = reinterpret_cast<uint8_t*>( image.data )[ src_idx+1 ];
                        pix[ dst_idx+2] = reinterpret_cast<uint8_t*>( image.data )[ src_idx+2 ];
                    }
                }
            } break;

            case BufferImageFormat::FLOAT3:
            {
                for( int j = height - 1; j >= 0; --j )
                {
                    for( int i = 0; i < width; ++i )
                    {
                        const int32_t dst_idx = 3*width*(height-j-1) + 3*i;
                        const int32_t src_idx = 3*width*j            + 3*i;
                        for( int elem = 0; elem < 3; ++elem )
                        {
                            const float   f = reinterpret_cast<float*>( image.data )[src_idx+elem ];
                            const int32_t v = static_cast<int32_t>( 256.0f*(disable_srgb_conversion ? f : toSRGB(f)) );
                            const int32_t c =  v < 0 ? 0 : v > 0xff ? 0xff : v;
                            pix[ dst_idx+elem ] = static_cast<uint8_t>( c );
                        }
                    }
                }
            } break;

            case BufferImageFormat::FLOAT4:
            {
                for( int j = height - 1; j >= 0; --j )
                {
                    for( int i = 0; i < width; ++i )
                    {
                        const int32_t dst_idx = 3*width*(height-j-1) + 3*i;
                        const int32_t src_idx = 4*width*j            + 4*i;
                        for( int elem = 0; elem < 3; ++elem )
                        {
                            const float   f = reinterpret_cast<float*>( image.data )[src_idx+elem ];
                            const int32_t v = static_cast<int32_t>( 256.0f*(disable_srgb_conversion ? f : toSRGB(f)) );
                            const int32_t c =  v < 0 ? 0 : v > 0xff ? 0xff : v;
                            pix[ dst_idx+elem ] = static_cast<uint8_t>( c );
                        }
                    }
                }
            } break;

            default:
            {
                throw Exception( "sutil::saveImage(): Unrecognized image buffer pixel format.\n" );
            }
        }

        savePPM( pix.data(), filename.c_str(), width, height, 3 );
    } */

    if(  ext == "PNG" || ext == "png" )
    {
        switch( image.pixel_format )
        {
            case BufferImageFormat::UNSIGNED_BYTE4:
            {
                stbi_flip_vertically_on_write( true );
                if( !stbi_write_png(
                            filename.c_str(),
                            image.width,
                            image.height,
                            4, // components,
                            image.data,
                            image.width*sizeof( uchar4 ) //stride_in_bytes
                            ) )
                    throw Exception( "sutil::saveImage(): stbi_write_png failed" );
            } break;

            case BufferImageFormat::FLOAT3:
            {
                throw Exception( "sutil::saveImage(): saving of float3 images to PNG not implemented yet" );
            }

            case BufferImageFormat::FLOAT4:
            {
                throw Exception( "sutil::saveImage(): saving of float4 images to PNG not implemented yet" );
            }

            default:
            {
                throw Exception( "sutil::saveImage: Unrecognized image buffer pixel format.\n" );
            }
        }
    }

    /* else if(  ext == "EXR" || ext == "exr" )
    {
        switch( image.pixel_format )
        {
            case BufferImageFormat::UNSIGNED_BYTE4:
            {
                throw Exception( "sutil::saveImage(): saving of uchar4 images to EXR not implemented yet" );
            }

            case BufferImageFormat::FLOAT3:
            {
                const char* err;
                int32_t ret = SaveEXR(
                        reinterpret_cast<float*>( image.data ),
                        image.width,
                        image.height,
                        3, // num components
                        static_cast<int32_t>( true ), // save_as_fp16
                        filename.c_str(),
                        &err );

                if( ret != TINYEXR_SUCCESS )
                    throw Exception( ( "sutil::saveImage( exr ) error: " + std::string( err ) ).c_str() );

            } break;

            case BufferImageFormat::FLOAT4:
            {
                const char* err;
                int32_t ret = SaveEXR(
                        reinterpret_cast<float*>( image.data ),
                        image.width,
                        image.height,
                        4, // num components
                        static_cast<int32_t>( true ), // save_as_fp16
                        filename.c_str(),
                        &err );

                if( ret != TINYEXR_SUCCESS )
                    throw Exception( ( "sutil::saveImage( exr ) error: " + std::string( err ) ).c_str() );
            } break;

            default:
            {
                throw Exception( "sutil::saveImage: Unrecognized image buffer pixel format.\n" );
            }
        }
    } */
    else
    {
        throw Exception( ( "sutil::saveImage(): Failed unsupported filetype '" + ext + "'" ).c_str() );
    }
}

}
