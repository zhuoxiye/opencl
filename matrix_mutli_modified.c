#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <CL/cl.h>
#include <time.h>


#define dim 1000
const char *ProgramSource = 
"__kernel void simpleMM(__global float* output,int widthA,int heightA,int widthB,int heightB,__global float* inputA,__global float* inputB,int TS){\n"
"float sum;\n"
"int col = get_local_id(0);\n"
"int row = get_local_id(1);\n  "
"int globalcol = TS * get_group_id(0) + col;\n"
"int globalrow = TS * get_group_id(1) + row;\n"
"int k,i;\n"
"__local float Asub[25 * 25];\n"
"__local float Bsub[25 * 25];\n"
"const int numtiles = widthB/TS;\n"
"//calculation\n"
"     sum = 0.0f;\n"
"     for(k = 0;k<numtiles;k++){\n"
"         const int tilerow = TS*k+row;\n  "
"         const int tilecol = TS*k+col;\n   "
"         Asub[row * TS + col] = inputA[globalrow*widthA + tilecol]; \n   "
"         Bsub[row * TS + col] = inputB[tilerow*widthB + globalcol]; \n         "
"     barrier(CLK_LOCAL_MEM_FENCE);\n"
"     for(i = 0;i<TS;i++){\n                            "
"         sum += Asub[row * TS + i] * Bsub[i * TS + col];\n"
"               }\n "
"     barrier(CLK_LOCAL_MEM_FENCE);\n                   "
"     }\n                                                  "
"      output[globalrow*widthB+globalcol] = sum; \n "
"}\n";

  cl_int wA=dim,hB=dim;
  cl_int hA=dim,wB=dim;
  


int main()
{

  clock_t start,finish;
  float *inputA;
  float *inputB;
  float *outputC;
  float *verify;
  
  inputA = (float *)malloc(wA*hA*sizeof(float));
  inputB = (float *)malloc(wB*hB*sizeof(float));
  outputC = (float *)malloc(hA*wB*sizeof(float));
  verify = (float *)malloc(hA*wB*sizeof(float));
   //initialization
   int i,j,k;
  
  for(i = 0;i<wA*hA;i++)
     {
      inputA[i] = rand()%100;
     }
  for(i = 0;i<wB*hB;i++)
     {
      inputB[i] = rand()%100;
     }
  for(i = 0; i<hA*wB ; i++)
    {
      verify[i] = 0;
    }

   start = clock();
  for(i = 0; i<hA ; i++)
    {
    for(j=0; j<wB ; j++)
      {
       for(k = 0; k<hB ; k++)
       verify[i*wB+j]+= inputA[k+i*wA]*inputB[j+k*wB];
      }
    }
  finish = clock();

  cl_int ciErrNum;
  cl_uint num_of_platforms=0;
  cl_uint num_of_devices=0;
  cl_platform_id* platform;
  cl_device_id* device;
  cl_device_id* device_cpu;
  cl_event prof_event;
  cl_ulong start_time = (cl_ulong)0;
  cl_ulong end_time = (cl_ulong)0;
  double run_time;


  //platforms
  ciErrNum = clGetPlatformIDs(5, NULL, &num_of_platforms);
  printf("num of platforms = %d\n",num_of_platforms);
  if(ciErrNum != CL_SUCCESS)
    {
      printf("unable to get platform_id\n");
      return 0;
    }
  platform = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_of_platforms); 
  ciErrNum = clGetPlatformIDs(num_of_platforms, platform,NULL);
  if(ciErrNum != CL_SUCCESS)
    {
	printf("unable to find any platforms\n");
    }

  //GPU device
  ciErrNum = clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_ALL, 1, NULL,&num_of_devices);
  device = (cl_device_id*)malloc(sizeof(cl_device_id) * num_of_devices);
  ciErrNum = clGetDeviceIDs(platform[1],CL_DEVICE_TYPE_ALL, num_of_devices,device,NULL);
  if(ciErrNum != CL_SUCCESS)
    {
      printf("unable to get device_id\n");
      return 0;
    }

   char name[48];
   ciErrNum =clGetDeviceInfo(device[0],CL_DEVICE_NAME,sizeof(name),name,NULL);
   if(ciErrNum != CL_SUCCESS)
     {
	printf("Couldn't read name data\n");
     }   
   printf("name:%s\n",name);

   size_t p;
   ciErrNum = clGetDeviceInfo(device[0],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(p),&p,NULL);
   printf("compute_units = %d\n ",p);
  
   
   cl_context_properties properties[]={CL_CONTEXT_PLATFORM,(cl_context_properties)platform[1],0};
   
  //create context
  cl_context ctx = clCreateContext(properties,1,&device[0],NULL,NULL,&ciErrNum);
 if(ciErrNum != CL_SUCCESS)
   {
	printf("unable to create context\n");
   }
 else
   {
     printf("succeed in creating context\n");
   }

  //create command queue
  cl_command_queue myqueue = clCreateCommandQueue(ctx,device[0],CL_QUEUE_PROFILING_ENABLE,&ciErrNum);
  if(ciErrNum != CL_SUCCESS)
    {
	printf("unable to create command_queue\n");
    }
  else
    {
      printf("succeed in creating command_queue\n");
    }
	  
//
  cl_mem bufferA = clCreateBuffer(ctx,CL_MEM_READ_ONLY,wA*hA*sizeof(float),NULL,&ciErrNum);
   if(ciErrNum != CL_SUCCESS)
    {
      printf("fail to create BufferA\n");
    }
  else
    {
      printf("succeed in creating BufferA\n");
    }

  ciErrNum = clEnqueueWriteBuffer(myqueue,bufferA,CL_TRUE,0,wA*hA*sizeof(float),inputA,0,NULL,NULL);
  if(ciErrNum != CL_SUCCESS)
    {
      printf("fail to enable BufferA\n");
    }
  else
    {
      printf("succeed in enabling BufferA\n");
    }
   cl_mem bufferB = clCreateBuffer(ctx,CL_MEM_READ_ONLY,wB*hB*sizeof(float),NULL,&ciErrNum);
 if(ciErrNum != CL_SUCCESS)
    {
      printf("fail to create BufferB\n");
    }
  else
    {
      printf("succeed in creating BufferB\n");
    }

  ciErrNum = clEnqueueWriteBuffer(myqueue,bufferB,CL_TRUE,0,wB*hB*sizeof(float),inputB,0,NULL,NULL);
 if(ciErrNum != CL_SUCCESS)
    {
      printf("fail to enable BufferB\n");
    }
  else
    {
      printf("succeed in enabling BufferB\n");
    }

  cl_mem bufferC = clCreateBuffer(ctx,CL_MEM_READ_ONLY,hA*wB*sizeof(float),NULL,&ciErrNum);
 if(ciErrNum != CL_SUCCESS)
    {
      printf("fail to create BufferC\n");
    }
  else
    {
      printf("succeed in creating BufferC\n");
    }
  
//
  cl_program myprog = clCreateProgramWithSource(ctx,1,(const char**)&ProgramSource,NULL,&ciErrNum);
  if(ciErrNum != CL_SUCCESS)
    {
	printf("fail to create program\n");
    }
  else
    {
      printf("succeed in creating program\n");
    }

  ciErrNum = clBuildProgram(myprog,0,NULL,NULL,NULL,NULL);
  if(ciErrNum != CL_SUCCESS)
    {
	size_t len;
	char buff[2048];
	printf("fail to build program\n");
	clGetProgramBuildInfo(myprog, device[0],CL_PROGRAM_BUILD_LOG,sizeof(buff),buff,&len);
	printf("%s\n",buff);
	
    }
  else
    {
      printf("succeed in building program\n");
    }

  //create kernel
  cl_kernel mykernel = clCreateKernel(myprog,"simpleMM",&ciErrNum);
  if(ciErrNum != CL_SUCCESS)
    {
	printf("fail to create kernel\n");
    }
  else
    {
      printf("succeed in creating kernel\n");
    }

  clSetKernelArg(mykernel,0,sizeof(cl_mem),&bufferC);
  clSetKernelArg(mykernel,1,sizeof(cl_int),(void*)&wA);
  clSetKernelArg(mykernel,2,sizeof(cl_int),(void*)&hA);
  clSetKernelArg(mykernel,3,sizeof(cl_int),(void*)&wB);
  clSetKernelArg(mykernel,4,sizeof(cl_int),(void*)&hB);
  clSetKernelArg(mykernel,5,sizeof(cl_mem),&bufferA);
  clSetKernelArg(mykernel,6,sizeof(cl_mem),&bufferB);

  const int TS = 25;
  clSetKernelArg(mykernel,7,sizeof(int),(void*)&TS);
  size_t localws[2] = {TS,TS};
  size_t globalws[2] = {hA,wA};
  
  ciErrNum = clEnqueueNDRangeKernel(myqueue,mykernel,2,NULL,globalws,localws,0,NULL,&prof_event);
  if(ciErrNum != CL_SUCCESS)
    {
	printf("Error queuing Kernel for execution\n");
    }
  else
    {
      printf("succeed in queuing kernel for execution\n");
    }
  
  clFinish(myqueue);
  ciErrNum = clWaitForEvents(1,&prof_event);
  if(ciErrNum != CL_SUCCESS)
    {
      printf("Error in clWaitForEvent\n");
    }
  
  ciErrNum = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),&start_time, NULL);
  ciErrNum = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&end_time,NULL);
  ciErrNum = clEnqueueReadBuffer(myqueue,bufferC,CL_TRUE,0,hA*wB*sizeof(float),(void*)outputC,0,NULL,NULL);

   if(ciErrNum != CL_SUCCESS)
     {
	printf("Error reading result buffer\n");
     }
   else
     {
       printf("succeed in reading result buffer\n");
     }
   
   run_time = (double)(end_time - start_time);
  


   //
   //////////////////////////////////////
   // verify result
   int flag=1;
   for(i = 0; i<hA*wB; i++)
     {
       if(outputC[i]!=verify[i])
	 flag = 0;
       break;
     }
   if(flag)
     printf("result is correct\n");
   else
     printf("result is not correct\n");

   printf("run time :%f s\n",run_time*1.0e-9);
   printf("%f \n",(double)(finish-start)/CLOCKS_PER_SEC);
   
   //////////////////////
   //realse
  clReleaseMemObject(bufferA);
  clReleaseMemObject(bufferB);
  clReleaseMemObject(bufferC);
  clReleaseProgram(myprog);
  clReleaseKernel(mykernel);
  clReleaseCommandQueue(myqueue);
  clReleaseContext(ctx);
  free(platform);
  free(device);
  free(inputA);
  free(inputB);
  free(outputC);
  free(verify);
  return 0;
}
