import { BuiltInProviderType } from 'next-auth/providers';
import { ClientSafeProvider, LiteralUnion, signIn } from 'next-auth/react';

const Login = ({providers}: {providers: Record<LiteralUnion<BuiltInProviderType, string>, ClientSafeProvider> | never[]}) => {
    return (
        <div className=" rounded-md p-4">
          <div className="w-full items-center justify-center h-screen flex flex-col bg-[#f5f8fa]">
            <div className="bg-white shadow-lg rounded-md p-4">
              <div className="w-full items-center justify-center flex">
                <div className="font-bold text-5xl w-full text-center mt-4 items-center justify-center flex">
                  {/* <img src={'/monthos.jpeg'} className="h-[80px]" /> */}
                  <div className=' font-bold text-sm md:text-[40px]'>Physio Tees</div>
                </div>
              </div>

              <div className="mt-10 flex flex-col gap-3 pb-4">
                {Object.values(providers).map((provider) => (
                  <div key={provider.name} className="">
                    {provider.name === 'Google' ? (
                      <div
                        onClick={() => {
                          signIn(provider.id);
                        }}
                        className="cursor-pointer hover:bg-[#e6644f] font-semibold text-[#e6644f] hover:text-white rounded-md border-[1px] w-[350px]"
                      >
                        <div className="flex flex-row rounded-md w-full">
                          <div className="w-[15%] border-r-[1px] p-3">
                            <svg
                              fill="#000000"
                              width="800px"
                              height="800px"
                              viewBox="0 0 32 32"
                              version="1.1"
                              xmlns="http://www.w3.org/2000/svg"
                              className="fill-current w-[15px] h-[15px]"
                            >
                              <title>google</title>
                              <path d="M16.601 14.648v4.105h9.813c-0.162 2.008-0.982 3.798-2.243 5.18l0.006-0.007c-1.825 1.86-4.364 3.012-7.172 3.012-0.142 0-0.283-0.003-0.424-0.009l0.020 0.001c-5.946-0.003-10.765-4.823-10.765-10.77 0-0.051 0-0.102 0.001-0.152l-0 0.008c-0.001-0.043-0.001-0.094-0.001-0.145 0-5.946 4.819-10.767 10.765-10.77h0c0.040-0.001 0.087-0.001 0.135-0.001 2.822 0 5.383 1.121 7.262 2.941l-0.003-0.003 2.888-2.888c-2.568-2.578-6.121-4.174-10.047-4.174-0.082 0-0.164 0.001-0.246 0.002l0.012-0c-0.002 0-0.005 0-0.008 0-8.337 0-15.11 6.699-15.228 15.009l-0 0.011c0.118 8.32 6.891 15.020 15.228 15.020 0.003 0 0.006 0 0.009 0h-0c0.169 0.007 0.367 0.012 0.566 0.012 3.892 0 7.407-1.616 9.91-4.213l0.004-0.004c2.208-2.408 3.561-5.63 3.561-9.169 0-0.15-0.002-0.3-0.007-0.449l0.001 0.022c0.001-0.054 0.001-0.119 0.001-0.183 0-0.844-0.079-1.669-0.231-2.469l0.013 0.082z"></path>
                            </svg>
                          </div>
                          <div className="ml-3 mx-10 w-[95%] flex items-center justify-center">
                            Sign in With Google
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div
                        onClick={() => {
                          signIn(provider.id);
                        }}
                        className="cursor-pointer hover:bg-[#027cca] font-semibold hover:text-white text-[#027cca] rounded-md border-[1px] w-[350px]"
                      >
                        <div className="flex flex-row rounded-md w-full">
                          <div className="w-[15%] border-r-[1px] p-3">
                            <svg
                              width="800px"
                              height="800px"
                              viewBox="0 0 24 24"
                              xmlns="http://www.w3.org/2000/svg"
                              className="fill-current w-[20px] h-[20px]"
                            >
                              <title>microsoft_windows</title>
                              <rect width="24" height="24" fill="none" />
                              <path d="M3,12V6.75L9,5.43v6.48L3,12M20,3v8.75L10,11.9V5.21L20,3M3,13l6,.09V19.9L3,18.75V13m17,.25V22L10,20.09v-7Z" />
                            </svg>
                          </div>
                          <div className="ml-3 mx-10 w-[95%] flex items-center justify-center">
                            Sign in With Microsoft
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
    )
}

export default Login;