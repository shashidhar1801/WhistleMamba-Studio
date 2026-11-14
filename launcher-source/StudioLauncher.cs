using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using System.Web.Script.Serialization;
using System.Collections.Generic;

[assembly:System.Reflection.AssemblyTitle("WhistleMamba Studio")]
[assembly:System.Reflection.AssemblyProduct("WhistleMamba Studio for Windows")]
[assembly:System.Reflection.AssemblyVersion("0.2.0.3")]

internal static class Studio {
    static readonly string root=AppDomain.CurrentDomain.BaseDirectory;
    static readonly string app=Path.Combine(root,"app");
    const string port="8785";
    static readonly string url="http://127.0.0.1:"+port+"/";
    static readonly string logs=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"WhistleMamba Studio","Logs");
    static string logfile,build;
    static Process worker;
    static NotifyIcon icon;
    static readonly object gate=new object();
    static readonly JavaScriptSerializer json=new JavaScriptSerializer();
    static bool quiet;
    static void Log(string line){if(line!=null) lock(gate){File.AppendAllText(logfile,DateTime.Now.ToString("s")+" "+line+Environment.NewLine);}}
    static void Alert(string text){Log(text);if(!quiet) MessageBox.Show(text+"\n\nLog: "+logfile,"WhistleMamba Studio",MessageBoxButtons.OK,MessageBoxIcon.Information);}
    static string LinuxPath(string path){
        string p=Path.GetFullPath(path);
        if(p.Length<3 || p[1]!=':' || p[2]!='\\') throw new Exception("Extract Studio onto a local Windows drive (not a network share).");
        return "/mnt/"+Char.ToLowerInvariant(p[0])+p.Substring(2).Replace('\\','/');
    }
    // No shell: quote one Windows command-line argument for CreateProcess.
    static string Arg(string a){if(a.Contains("\"") || a.EndsWith("\\") || a.Contains("\r") || a.Contains("\n")) throw new Exception("Unsupported character in a configured path.");return a.Length==0 || a.IndexOfAny(new[]{' ','\t'})>=0 ? "\""+a+"\"" : a;}
    static bool Ready(){
        try{
            var req=(HttpWebRequest)WebRequest.Create(url+"launcher-health");req.Proxy=null;req.Timeout=900;req.ReadWriteTimeout=900;req.AllowAutoRedirect=false;
            using(var resp=req.GetResponse())using(var reader=new StreamReader(resp.GetResponseStream())){
                var v=json.Deserialize<Dictionary<string,object>>(reader.ReadToEnd());
                return (string)v["service"]=="WhistleMamba Studio for Windows" && (string)v["studio_build"]==build && (bool)v["model"] && Convert.ToInt32(v["preview_seconds"])==15;
            }
        }catch{return false;}
    }
    static void Open(){
        foreach(var parent in new[]{Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86),Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData)}){
            var chrome=Path.Combine(parent,@"Google\Chrome\Application\chrome.exe");
            if(File.Exists(chrome)){Process.Start(new ProcessStartInfo(chrome,"--app="+url){UseShellExecute=false});return;}
        }
        Process.Start(new ProcessStartInfo(url){UseShellExecute=true});
    }
    static void StartRuntime(string mode){
        var cfg=json.Deserialize<Dictionary<string,string>>(File.ReadAllText(Path.Combine(root,"LocalRuntime.json")));
        string checkpoint=cfg["checkpoint_windows"];
        if(!File.Exists(checkpoint))throw new Exception("The selected model file was moved or removed. Run Configure Detector.exe and select it again.");
        var args=new List<string>{"-d",cfg["distribution"],"--cd",LinuxPath(app),"--exec","/usr/bin/env","PYTHONPATH=.","PYTHONDONTWRITEBYTECODE=1"};
        args.Add("WHISTLEMAMBA_WINDOWS_DESKTOP="+LinuxPath(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory)));
        args.Add("WHISTLEMAMBA_WINDOWS_DOCUMENTS="+LinuxPath(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments)));
        args.Add("WHISTLEMAMBA_WINDOWS_DOWNLOADS="+LinuxPath(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),"Downloads")));
        if(mode=="check-editor") args.Add("QT_QPA_PLATFORM=offscreen");
        args.AddRange(new[]{cfg["linux_python"],"-u","windows_entry.py",mode,LinuxPath(checkpoint),port});
        var pi=new ProcessStartInfo(Path.Combine(Environment.SystemDirectory,"wsl.exe"),String.Join(" ",args.ConvertAll(Arg).ToArray()));
        pi.CreateNoWindow=true;pi.WindowStyle=ProcessWindowStyle.Hidden;pi.UseShellExecute=false;pi.RedirectStandardOutput=true;pi.RedirectStandardError=true;pi.RedirectStandardInput=true;
        worker=new Process{StartInfo=pi};worker.OutputDataReceived+=(s,e)=>{Log(e.Data);if(e.Data=="STUDIO_BUSY" && icon!=null)icon.ShowBalloonTip(2500,"Analysis in progress","Finish the current recording before exiting.",ToolTipIcon.Info);};
        worker.ErrorDataReceived+=(s,e)=>Log(e.Data);worker.Start();worker.BeginOutputReadLine();worker.BeginErrorReadLine();Log("Started "+mode+" (WSL wrapper "+worker.Id+").");
    }
    static void StartEditor(bool check){
        var cfg=json.Deserialize<Dictionary<string,string>>(File.ReadAllText(Path.Combine(root,"LocalRuntime.json")));
        string native;
        if(!cfg.TryGetValue("windows_gui_python",out native)){StartRuntime(check?"check-editor":"desktop");return;}
        if(!File.Exists(native))throw new Exception("The configured Windows GUI environment is missing.");
        var pi=new ProcessStartInfo(native, "-u "+Arg(Path.Combine(app,"windows_entry.py"))+" "+(check?"check-editor":"desktop")+" "+Arg(cfg["checkpoint_windows"])+" 8772");
        pi.WorkingDirectory=app;pi.UseShellExecute=false;pi.CreateNoWindow=true;
        pi.RedirectStandardOutput=true;pi.RedirectStandardError=true;
        pi.EnvironmentVariables["PYTHONPATH"]=app;pi.EnvironmentVariables["PYTHONDONTWRITEBYTECODE"]="1";
        pi.EnvironmentVariables["PYTHONIOENCODING"]="utf-8";
        pi.EnvironmentVariables["QT_QPA_PLATFORM"]=check?"offscreen":"windows";
        pi.EnvironmentVariables["WHISTLEMAMBA_WINDOWS_DESKTOP"]=Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
        pi.EnvironmentVariables["WHISTLEMAMBA_WINDOWS_DOCUMENTS"]=Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
        worker=new Process{StartInfo=pi};worker.OutputDataReceived+=(s,e)=>Log(e.Data);worker.ErrorDataReceived+=(s,e)=>Log(e.Data);
        worker.Start();worker.BeginOutputReadLine();worker.BeginErrorReadLine();Log("Started native Windows Qt "+(check?"self-test":"editor")+" (PID "+worker.Id+"). CUDA analysis stays in WSL.");
    }
    static void Stop(){if(worker!=null && !worker.HasExited){worker.StandardInput.WriteLine("STOP");worker.StandardInput.Flush();}}
    static void Tray(){
        using(var menu=new ContextMenuStrip())using(icon=new NotifyIcon())using(var timer=new System.Windows.Forms.Timer()){
            menu.Items.Add("Open WhistleMamba Studio",null,(s,e)=>Open());
            menu.Items.Add("Exit Studio…",null,(s,e)=>{if(MessageBox.Show("Download any results you want to keep first. Exit this local session? Your original WAVs will not be changed.","Exit WhistleMamba Studio",MessageBoxButtons.YesNo,MessageBoxIcon.Question,MessageBoxDefaultButton.Button2)==DialogResult.Yes)Stop();});
            icon.Icon=Icon.ExtractAssociatedIcon(Application.ExecutablePath);icon.Text="WhistleMamba Studio";icon.ContextMenuStrip=menu;icon.DoubleClick+=(s,e)=>Open();icon.Visible=true;
            timer.Interval=1000;timer.Tick+=(s,e)=>{if(worker.HasExited)Application.ExitThread();};timer.Start();Application.Run();icon.Visible=false;
        }
    }
    [STAThread] static int Main(string[] args){
        bool pipeTest=args.Length==1 && args[0]=="--example-session";
        quiet=Array.Exists(args,a=>a=="--check" || a=="--check-editor" || a=="--start-only") || pipeTest;
        try{
            Directory.CreateDirectory(logs);logfile=Path.Combine(logs,DateTime.Now.ToString("yyyyMMdd-HHmmss-fff")+"-"+Process.GetCurrentProcess().Id+".log");
            build=File.ReadAllText(Path.Combine(app,"BUILD-ID")).Trim();
            if(!File.Exists(Path.Combine(root,"LocalRuntime.json"))){
                Alert("First run Configure Detector.exe from this folder. Select the CSR01 inference model and choose Verify and save model.");return 1;
            }
            bool check=Array.IndexOf(args,"--check")>=0;
            bool editorCheck=Array.IndexOf(args,"--check-editor")>=0;
#if DESKTOP
            StartEditor(editorCheck);worker.WaitForExit();if(worker.ExitCode!=0)Alert("The desktop editor did not start correctly. Check the local runtime and log.");return worker.ExitCode;
#else
            bool owner=false;
            using(var mutex=new Mutex(true,@"Local\WhistleMambaStudioWindows"+port,out owner)){
                if(pipeTest && (!owner || Ready()))throw new Exception("A Studio session is already open. Test did not start or change it.");
                if(owner && !Ready()) StartRuntime("web");
                DateTime end=DateTime.UtcNow.AddSeconds(55);
                while(!Ready()){
                    if(worker!=null && worker.HasExited) throw new Exception("Studio could not start its local server. Check the log and local port "+port+".");
                    if(DateTime.UtcNow>end){if(worker!=null)worker.StandardInput.Close();throw new Exception("Studio did not become ready. Check the configured runtime and local port "+port+".");}
                    Thread.Sleep(250);
                }
                Log("Ready: verified build "+build+" at "+url);
                if(pipeTest){try{Console.WriteLine("STUDIO_READY");Console.Out.Flush();Console.ReadLine();}finally{Stop();}
                    if(!worker.WaitForExit(60000))throw new Exception("Example service did not exit.");return worker.ExitCode;}
                if(check){if(worker!=null){Stop();if(!worker.WaitForExit(10000))throw new Exception("Service did not close after self-test.");}return 0;}
                if(!quiet)Open();
                if(worker!=null)Tray();
                return 0;
            }
#endif
        }catch(Exception e){if(logfile!=null)Alert(e.Message);else if(!quiet)MessageBox.Show(e.Message,"WhistleMamba Studio");return 1;}
    }
}
